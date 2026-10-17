"""Frozen online encoders for the actual dynamically sampled training inputs."""
import gc
import json
from asset_paths import loads as asset_loads
from pathlib import Path

import torch
from torch.nn import functional as F
from PIL import Image

from klein_style_condition import klein_prompt
from prepare_char_klein import caption_tokens
from story_klein_mapper import mapper_messages


def fit_reference(image):
    image = image.convert('RGB')
    image.thumbnail((256, 256), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (256, 256), (128, 128, 128))
    canvas.paste(image, ((256-image.width)//2, (256-image.height)//2))
    return canvas


def teacher_bindings(tokenizer, teacher):
    words, ends = [], []
    for word in teacher['marked_caption'].split():
        if word == '<|image|>':
            ends.append(len(' '.join(words)))
        else:
            words.append(word)
    if len(ends) != len(teacher['references']):
        raise ValueError('Released teacher markers and bank references differ')
    prompt = klein_prompt(' '.join(words))
    _, encoded, start = caption_tokens(tokenizer, prompt, 128)
    if len(encoded['input_ids']) > 128:
        raise ValueError('Refusing teacher prompt truncation')
    positions = []
    for end in ends:
        locations = [i for i, (a, b) in enumerate(encoded['offset_mapping']) if b > a and a < start+end <= b]
        if not locations:
            raise ValueError('Teacher marker end does not bind to a native token')
        positions.append(locations[-1])
    if len(set(positions)) != len(positions):
        raise ValueError('Ambiguous teacher binding')
    return prompt, positions


class OnlineEncoders:
    def __init__(self, model_path, qwen_path, bundle_root, device, stage):
        from diffusers import Flux2KleinPipeline
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        from qwen_patch_embedding import install_linear_patch
        self.device, self.stage = device, stage
        self.pipe = Flux2KleinPipeline.from_pretrained(model_path, transformer=None, torch_dtype=torch.bfloat16,
                                                      local_files_only=True).to(device)
        self.pipe.text_encoder.eval().requires_grad_(False)
        self.pipe.vae.eval().requires_grad_(False)
        self.pipe.set_progress_bar_config(disable=True)
        self.processor = AutoProcessor.from_pretrained(qwen_path, local_files_only=True)
        self.processor.tokenizer.padding_side = 'left'
        self.processor.image_processor.size = {'longest_edge': 65536, 'shortest_edge': 65536}
        model = Qwen3_5ForConditionalGeneration.from_pretrained(qwen_path, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 'cpu'}).eval().requires_grad_(False)
        self.patch = install_linear_patch(model)
        self.counts = {'qwen': sum(p.numel() for p in model.parameters()),
                      'text_encoder': sum(p.numel() for p in self.pipe.text_encoder.parameters()),
                      'vae': sum(p.numel() for p in self.pipe.vae.parameters())}
        self.visual = model.model.visual
        if stage == 'story':
            self.qwen = model.to(device)
        else:
            self.qwen = None
            del model
            gc.collect()
            self.visual.to(device)
        self.bank_rows = asset_loads((Path(bundle_root)/'reference_bank.json').read_text())
        images = []
        for row in self.bank_rows:
            with Image.open(row['path']) as im:
                images.append(fit_reference(im))
        self.bank_features = self.vision(images)
        self.bank_indices = {row['name']: i for i, row in enumerate(self.bank_rows)}

    @torch.no_grad()
    def vision(self, images):
        if not images:
            return torch.zeros(0, 2560, dtype=torch.bfloat16, device=self.device)
        output = []
        for start in range(0, len(images), 16):
            values = self.processor.image_processor(images=images[start:start+16], return_tensors='pt').to(self.device)
            result = self.visual(values['pixel_values'].to(torch.bfloat16), grid_thw=values['image_grid_thw']).pooler_output
            sizes = (values['image_grid_thw'].prod(-1)//self.visual.spatial_merge_size**2).tolist()
            output.extend(x.float().mean(0).to(torch.bfloat16) for x in result.split(sizes))
        return torch.stack(output).contiguous()

    @torch.no_grad()
    def text(self, prompts):
        for prompt in prompts:
            if len(caption_tokens(self.pipe.tokenizer, prompt)[1]['input_ids']) > 128:
                raise ValueError('Refusing prompt truncation')
        text, ids = self.pipe.encode_prompt(prompts, device=self.device, max_sequence_length=128)
        return text, ids.to(self.device)

    @torch.no_grad()
    def image(self, pixels):
        latents = self.pipe._encode_vae_image(torch.stack(pixels).to(self.device, dtype=torch.bfloat16), None)
        return self.pipe._pack_latents(latents), self.pipe._prepare_latent_ids(latents).to(self.device)

    @torch.no_grad()
    def character(self, records):
        text, ids = self.text([r['prompt'] for r in records])
        target, image_ids = self.image([r['pixels'] for r in records])
        if any('reference_bank_indices' in row for row in records):
            from character_reference_training import mixed_visual_features
            features = mixed_visual_features(records, self.vision, self.bank_features)
        else:
            features = self.vision([im for r in records for im in r['references']])
        visual, cursor = [], 0
        for row in records:
            n = len(row['references'])
            visual.append(F.pad(features[cursor:cursor+n], (0, 0, 0, 4-n)))
            cursor += n
        return {'target': target, 'text': text, 'text_ids': ids[0], 'image_ids': image_ids[0],
                'visual': torch.stack(visual),
                'indices': torch.stack([F.pad(r['indices'], (0, 4-len(r['indices'])), value=-1) for r in records]).to(self.device),
                'masks': torch.stack([F.pad(r['masks'], (0, 0, 0, 4-len(r['indices']))) for r in records]).to(self.device),
                'foreground': torch.stack([r['foreground'] for r in records]).to(self.device)}

    @torch.no_grad()
    def context(self, inputs):
        prompts, images = [], []
        for row in inputs:
            prompts.append(self.processor.apply_chat_template(mapper_messages(row['history'], row['current_caption']),
                           tokenize=False, add_generation_prompt=True, enable_thinking=False))
            images.extend(h['image'] for h in row['history'])
        values = self.processor(text=prompts, images=images, padding=True, return_tensors='pt').to(self.device)
        values['pixel_values'] = values['pixel_values'].to(torch.bfloat16)
        if len(values['image_grid_thw']) != len(images) or not bool(values['attention_mask'][:, -8:].all()):
            raise ValueError('Causal input image/state coverage differs')
        return self.qwen.model(**values, use_cache=False, return_dict=True).last_hidden_state[:, -8:].contiguous()

    @torch.no_grad()
    def story(self, records):
        model_inputs = [r['model_inputs'] for r in records]
        current, text_ids = self.text([klein_prompt(r['current_caption']) for r in model_inputs])
        qwen_states = self.context(model_inputs)
        teachers = [r['supervision']['teacher'] for r in records]
        bound = [teacher_bindings(self.pipe.tokenizer, t) for t in teachers]
        prompts = [x[0] for x in bound]
        teacher_text, _ = self.text(prompts)
        indices, visual = [], []
        for teacher, (_, positions) in zip(teachers, bound):
            features = []
            for reference in teacher['references']:
                name = reference['name']
                expected = self.bank_rows[self.bank_indices[name]]['path']
                if reference['image'] != expected:
                    raise ValueError('Teacher reference differs from explicit fixed bank')
                features.append(self.bank_features[self.bank_indices[name]])
            indices.append(torch.tensor(positions, dtype=torch.int64, device=self.device))
            visual.append(torch.stack(features) if features else teacher_text.new_zeros((0, 2560)))
        width = max(1, max(len(x) for x in indices))
        target, image_ids = self.image([r['supervision']['pixels'] for r in records])
        return {'target': target, 'image_ids': image_ids[0], 'text_ids': text_ids[0], 'current': current,
                'qwen_states': qwen_states, 'teacher_text': teacher_text,
                'indices': torch.stack([F.pad(x, (0, width-len(x)), value=-1) for x in indices]),
                'visual': torch.stack([F.pad(x, (0, 0, 0, width-len(x))) for x in visual])}
