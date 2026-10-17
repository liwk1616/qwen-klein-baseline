"""Dynamic StoryGPT-V samples with explicit visits and causal supervision.

No frozen frame cache is read. Sampling is a function of an explicit visit
number, so restarting an optimizer checkpoint selects the same next images.
"""
from collections import Counter
from copy import deepcopy
import random
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from storygptv_data_bundle import StoryGPTVDataBundle, draw_stage2_metadata
from prepare_char_klein import caption_tokens
from klein_style_condition import klein_prompt


class VisitSampler:
    def __init__(self, count, rank, world, seed):
        if count < 1 or not 0 <= rank < world:
            raise ValueError('Invalid sample population or rank')
        self.count, self.rank, self.world, self.seed = count, rank, world, seed
        self.epoch, self.order = None, None

    def index(self, visit):
        # Interleave ranks across one continuous sequence of full permutations.
        # This also handles incomplete final rank groups without padding or loss.
        # For divisible populations the original per-rank order is unchanged.
        epoch, offset = divmod(visit * self.world + self.rank, self.count)
        if epoch != self.epoch:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            self.order = torch.randperm(self.count, generator=generator).tolist()
            self.epoch = epoch
        return self.order[offset]


class LayoutSampler:
    """Repartition the same optimizer-step inputs when physical GPU count changes."""
    def __init__(self, count, rank, world, batch, accumulation, seed, logical):
        self.rank, self.world, self.batch = rank, world, batch
        self.accumulation, self.seed, self.logical = accumulation, seed, logical
        if world * batch * accumulation != logical['world_size'] * logical['batch_size'] * logical['accumulation']:
            raise ValueError('Sampler migration must preserve the global optimizer batch')
        self.samplers = [VisitSampler(count, r, logical['world_size'], seed)
                         for r in range(logical['world_size'])]

    def coordinates(self, visit):
        step, within = divmod(visit, self.batch * self.accumulation)
        micro, item = divmod(within, self.batch)
        global_item = micro * self.world * self.batch + self.rank * self.batch + item
        old_micro, remainder = divmod(global_item, self.logical['world_size'] * self.logical['batch_size'])
        old_rank, old_item = divmod(remainder, self.logical['batch_size'])
        old_visit = step * self.logical['batch_size'] * self.logical['accumulation'] + old_micro * self.logical['batch_size'] + old_item
        return old_rank, old_visit

    def sample(self, visit):
        logical_rank, logical_visit = self.coordinates(visit)
        return (self.samplers[logical_rank].index(logical_visit),
                sample_seed(self.seed, logical_rank, logical_visit), logical_rank, logical_visit)


def sample_seed(seed, rank, visit):
    return seed + rank * 1000000007 + visit * 1009


def pooled_mask(mask):
    return mask.float().reshape(32, 16, 32, 16).mean((1, 3)).reshape(1024)


def bind_segments(tokenizer, tokens, segments):
    """Bind annotation end offsets, including repeated mentions, to Klein tokens."""
    caption = ' '.join(tokens)
    prompt = klein_prompt(caption)
    _, encoded, start = caption_tokens(tokenizer, prompt, 128)
    if len(encoded['input_ids']) > 128:
        raise ValueError('Refusing native caption truncation')
    result = []
    for item in segments:
        end = int(item['end'])
        begin = int(item['start'])
        if not 0 <= begin < end <= len(tokens):
            raise ValueError('Invalid source annotation token span')
        end_char = start + len(' '.join(tokens[:end]))
        locations = [i for i, (a, b) in enumerate(encoded['offset_mapping']) if b > a and a < end_char <= b]
        if not locations:
            raise ValueError('Annotation end is absent from native token offsets')
        result.append(dict(item, token_index=locations[-1]))
    counts = Counter(x['token_index'] for x in result)
    return prompt, [x for x in result if counts[x['token_index']] == 1]


class AlignedData:
    def __init__(self, root, tokenizer):
        self.bundle = StoryGPTVDataBundle(root)
        self.tokenizer = tokenizer
        # This source file was individually read during the protocol audit. It
        # contains CPU image transforms only, no model/network initialization.
        path = Path(__file__).with_name('public_transforms.py')
        scope = {'__name__': 'reviewed_storygptv_transforms'}
        exec(compile(path.read_text(), str(path), 'exec'), scope)
        args = SimpleNamespace(train_resolution=512, object_resolution=256,
                               no_object_augmentation=False, object_background_processor='random')
        self.frame_transform = scope['get_train_transforms_with_segmap'](args)
        self.object_transform = scope['get_object_transforms'](args)
        self.object_processor = scope['get_object_processor'](args)

    def character(self, index, seed):
        rng = random.Random(seed)
        bundle = self.bundle
        row = bundle.by_split['train'][index]
        ann = bundle.annotations[row['clip_id']]
        frame = rng.randrange(row['frame_count'])
        probability = rng.random()
        # Follow intended unconditional/text-only/full conditioning. The
        # released empty-caption/token reconstruction bug is not replicated.
        kind = 'unconditional' if probability < .1 else 'text_only' if probability < .2 else 'character_augmented'
        segments = deepcopy(ann['segments']) if kind == 'character_augmented' else []
        if len(segments) > 4:
            segments = rng.sample(segments, 4)
        segments.sort(key=lambda x: x['end'])
        prompt, objects = bind_segments(self.tokenizer, ann['tokens'], segments)
        if kind == 'unconditional':
            prompt = ''
        image = torch.from_numpy(bundle.image(row['clip_id'], frame)).permute(2, 0, 1)
        segmentation, present = bundle.mask(row['clip_id'], frame)
        if not present:
            raise ValueError('A required audited training mask is missing')
        segmentation = torch.from_numpy(np.array(segmentation, copy=True))
        # CPU augmentation has a separate visit seed and cannot alter CUDA
        # diffusion RNG. fork_rng restores the caller's CPU state on exit.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
            pixels, transformed_mask = self.frame_transform(image, segmentation)
            background = self.object_processor.get_background(image)
            references, masks, kept = [], [], []
            for item in objects:
                if not bool((segmentation == item['id']).any()):
                    continue
                x1, y1, x2, y2 = [int(v) * 4 for v in item['bbox']]
                bbox = [max(0, x1), max(0, y1), min(512, x2), min(512, y2)]
                if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                    continue
                crop = self.object_processor(image.clone(), background, segmentation, item['id'], bbox)
                crop = self.object_transform(crop).clamp(0, 1)
                references.append(Image.fromarray((crop.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)))
                masks.append(pooled_mask(transformed_mask == item['id']))
                kept.append(dict(item, bbox_xyxy=bbox))
        meta = {'clip_id': row['clip_id'], 'split': 'train', 'frame_index': frame,
                'sample_seed': seed, 'condition_kind': kind, 'prompt': prompt,
                'mask_member': row['mask_members'][frame], 'objects': kept,
                'omitted_invisible_invalid_or_ambiguous_segments': len(segments) - len(kept)}
        return {'pixels': pixels, 'prompt': prompt, 'references': references,
                'indices': torch.tensor([x['token_index'] for x in kept], dtype=torch.int64),
                'masks': torch.stack(masks) if masks else torch.zeros(0, 1024),
                'foreground': pooled_mask(transformed_mask != 0), 'metadata': meta}

    def story(self, index, seed):
        bundle = self.bundle
        source = bundle.stories['train'][index]
        story = dict(source, train_captions=[t['caption'] for t in source['evaluation_targets']])
        draw = draw_stage2_metadata(story, bundle.clips, random.Random(seed))
        target = draw['target']
        history = [dict(h, image=Image.fromarray(bundle.image(h['clip_id'], h['frame_index']))) for h in draw['history']]
        if any(h['clip_id'] == target['clip_id'] for h in history) or len(history) != draw['target_position'] - 1:
            raise ValueError('Noncausal story history')
        target_pixels = torch.from_numpy(bundle.image(target['clip_id'], target['frame_index'])).permute(2, 0, 1)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
            # Public image reconstruction transforms horizontally flip only
            # the target; Qwen history remains chronological source images.
            target_pixels, _ = self.frame_transform(target_pixels, torch.zeros(512, 512, dtype=torch.uint8))
        teacher = bundle.teacher[target['clip_id']]
        if teacher['reference_kind'] != 'official_fixed_character_bank':
            raise ValueError('Unexpected teacher reference policy')
        return {'model_inputs': {'history': history, 'current_caption': draw['current_caption']},
                'supervision': {'pixels': target_pixels, 'teacher': teacher},
                'metadata': {**draw, 'sample_seed': seed, 'caption_profile': 'official_evaluation_rule',
                             'target_image_in_model_inputs': False, 'teacher_in_model_inputs': False}}
