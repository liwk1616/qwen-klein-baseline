"""Raw data and sampling for the released StoryGPT-V Flintstones protocol.

This loader returns images and supervision before model-specific transforms.
It deliberately does not reuse the old fixed-frame Klein feature caches.
Archive members are decoded by archive_io without archive integrity operations.
"""
from __future__ import annotations

from copy import deepcopy
from io import BytesIO
import json
from asset_paths import loads as asset_loads
from pathlib import Path
import random
import zipfile

import numpy as np

from archive_io import zip_payload
from storygptv_protocol import causal_history


def read_jsonl(path):
    with Path(path).open() as stream:
        return [asset_loads(line) for line in stream if line.strip()]


def draw_stage1_metadata(row, annotation, rng):
    """Match ImageDataset's frame/condition/object sampling before transforms."""
    frame_index = rng.randrange(row['frame_count'])
    segments = deepcopy(annotation['segments'])
    probability = rng.random()
    # The released code only applies text-only dropout when <= 1 object exists.
    if probability < 0.1:
        condition_kind, segments = 'unconditional', []
    elif len(segments) <= 1 and probability < 0.2:
        condition_kind, segments = 'text_only', []
    else:
        condition_kind = 'character_augmented'
    if len(segments) > 4:
        segments = rng.sample(segments, 4)
    segments.sort(key=lambda item: item['end'])
    return {
        'clip_id': row['clip_id'], 'frame_index': frame_index,
        'condition_kind': condition_kind,
        'caption': '' if condition_kind == 'unconditional' else annotation['caption'],
        # Retain source tokens: the released tokenizer inserts markers into this
        # list even when its separate caption variable is empty.
        'source_tokens': list(annotation['tokens']), 'segments': segments,
        'mask_member': row['mask_members'][frame_index],
    }


def draw_stage2_metadata(story, clips, rng):
    """Match gill/data.py: draw four frames, then draw target index 1, 2 or 3."""
    frame_indices = [rng.randrange(clips[c]['frame_count']) for c in story['clip_ids']]
    target_index = rng.sample(range(1, len(story['clip_ids'])), 1)[0]
    return {
        'story_id': story['story_id'], 'target_position': target_index + 1,
        'sampled_frame_indices': frame_indices,
        'history': [
            {'clip_id': clip_id, 'frame_index': frame_indices[i],
             'caption': story['train_captions'][i]}
            for i, clip_id in enumerate(story['clip_ids'][:target_index])
        ],
        'current_caption': story['train_captions'][target_index],
        'target': {'clip_id': story['clip_ids'][target_index],
                   'frame_index': frame_indices[target_index]},
    }


class StoryGPTVDataBundle:
    """CPU loader; image arrays and target supervision stay explicitly separate.

    Model-specific augmentation, tokenization and feature encoding are not
    performed here. See protocol.json for the released transform configuration.
    Pass a persistent Random instance from a training worker so repeated visits
    draw new frames/targets; its state belongs in the training checkpoint.
    """

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.protocol = asset_loads((self.root / 'protocol.json').read_text())
        self.by_split = {
            mode: read_jsonl(self.root / 'clips' / f'{mode}.jsonl')
            for mode in ('train', 'val', 'test')
        }
        self.clips = {row['clip_id']: row for rows in self.by_split.values() for row in rows}
        self.stories = {
            mode: read_jsonl(self.root / 'stories' / f'{mode}.jsonl')
            for mode in ('train', 'val', 'test')
        }
        self.teacher = {
            row['clip_id']: row for row in read_jsonl(self.root / 'teacher_conditioning.jsonl')
        }
        self.annotations = asset_loads(Path(self.protocol['assets']['annotations']).read_text())
        self.archive_path = Path(self.protocol['assets']['mask_archive'])
        self._archive = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_archive'] = None
        return state

    def close(self):
        if self._archive is not None:
            self._archive.close()
            self._archive = None

    def image(self, clip_id, frame_index):
        array = np.load(self.clips[clip_id]['image_array'], mmap_mode='r', allow_pickle=False)
        if not 0 <= frame_index < array.shape[0]:
            raise IndexError((clip_id, frame_index))
        return np.array(array[frame_index], copy=True)

    def mask(self, clip_id, frame_index):
        if self._archive is None:
            self._archive = zipfile.ZipFile(self.archive_path)
        row = self.clips[clip_id]
        member = row['mask_members'][frame_index]
        try:
            entry = self._archive.getinfo(member)
        except KeyError:
            # Public ImageDataset falls back to an all-zero segmentation map.
            return np.zeros(row['image_shape'][:2], dtype=np.int64), False
        payload = b''.join(zip_payload(self.archive_path, entry))
        array = np.load(BytesIO(payload), allow_pickle=False)
        if list(array.shape) != row['image_shape'][:2]:
            raise ValueError(f'Unexpected mask shape: {member}: {array.shape}')
        return array, True

    def stage1(self, index, rng: random.Random, split='train'):
        row = self.by_split[split][index]
        draw = draw_stage1_metadata(row, self.annotations[row['clip_id']], rng)
        segmentation, mask_present = self.mask(row['clip_id'], draw['frame_index'])
        return {
            'sampling': draw,
            'image': self.image(row['clip_id'], draw['frame_index']),
            'segmentation': segmentation,
            'mask_present': mask_present,
            'source_annotation': deepcopy(self.annotations[row['clip_id']]),
        }

    def stage2(self, index, rng: random.Random, split='train'):
        story = self.stories[split][index]
        draw = draw_stage2_metadata(story, self.clips, rng)
        target = draw['target']
        # The source draws frames for all four clips, but only the causal prefix
        # is allowed into model_inputs. Future images are not even opened here.
        return {
            'sampling': draw,
            'model_inputs': {
                'history': [dict(item, image=self.image(item['clip_id'], item['frame_index']))
                            for item in draw['history']],
                'current_caption': draw['current_caption'],
            },
            'supervision': {
                'clip_id': target['clip_id'],
                'target_image': self.image(target['clip_id'], target['frame_index']),
                'teacher_conditioning': deepcopy(self.teacher[target['clip_id']]),
            },
        }

    def evaluation_step(self, index, position, completed, split='test'):
        story = self.stories[split][index]
        targets = story['evaluation_targets']
        result = causal_history(story['story_id'], targets, completed, position)
        result['reference_bank_names'] = (
            list(targets[0]['active_reference_bank_names']) if position == 0 else []
        )
        return result
