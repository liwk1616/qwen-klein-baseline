"""Explicit fixed-bank choices without consuming any existing training RNG."""
from collections import Counter

import torch


ALIASES = {name: name for name in ('fred', 'wilma', 'barney', 'betty', 'pebbles', 'mr slate', 'dino')}
ALIASES.update({'mr. slate': 'mr slate', 'mr . slate': 'mr slate'})


def canonical_name(value):
    return ALIASES.get(' '.join(value.casefold().split()))


class ReferenceMixer:
    def __init__(self, bank_rows, probability, seed, rank=0):
        if not 0 <= probability <= 1:
            raise ValueError('Invalid fixed-bank probability')
        self.bank = {row['name']: (index, row['path']) for index, row in enumerate(bank_rows)}
        if set(self.bank) != set(ALIASES.values()) or len(bank_rows) != 7:
            raise ValueError('Expected the seven explicitly named official references')
        self.probability = probability
        self.rng = torch.Generator().manual_seed(seed + rank * 1009)

    def apply(self, sample, annotation, counts):
        objects = sample['metadata']['objects']
        if len(objects) != len(sample['references']) or len(objects) != len(sample['indices']):
            raise ValueError('Reference/object/token layout differs')
        if objects and sample['metadata']['condition_kind'] != 'character_augmented':
            raise ValueError('References present in a dropped character condition')
        identities_by_id = {}
        for obj in annotation['segments']:
            name = canonical_name(obj['word'])
            if name:
                identities_by_id.setdefault(obj['id'], set()).add(name)
        decisions, selected, records = {}, [], []
        increment = Counter()
        for obj in objects:
            span = ' '.join(annotation['tokens'][obj['start']:obj['end']])
            name = canonical_name(obj['word'])
            reason = 'eligible'
            if name is None:
                reason = 'unrecognized_or_generic_identity'
            elif canonical_name(span) != name:
                reason = 'annotation_word_and_token_span_disagree'
            elif len(identities_by_id[obj['id']]) != 1:
                reason = 'ambiguous_segment_identity'
            eligible = reason == 'eligible'
            if eligible and name not in decisions:
                decisions[name] = bool(torch.rand((), generator=self.rng) < self.probability)
                increment['eligible_distinct_identity_visits'] += 1
                increment['bank_distinct_identity_visits'] += int(decisions[name])
            bank = eligible and decisions[name]
            index, path = self.bank[name] if bank else (-1, None)
            selected.append(index)
            records.append({'segment_id': obj['id'], 'word': obj['word'], 'source_span': span,
                            'token_index': obj['token_index'], 'canonical_name': name,
                            'source': 'official_fixed_bank' if bank else 'original_crop',
                            'bank_index': index, 'bank_image': path, 'eligibility': reason})
            increment['bound_mentions'] += 1
            increment['bank_mentions' if bank else 'crop_mentions'] += 1
            if eligible:
                increment['eligible_mentions'] += 1
                increment[f'{name}:bank' if bank else f'{name}:crop'] += 1
            else:
                increment['skipped:' + reason] += 1
        increment['sample_visits'] += 1
        increment['frames_with_bank'] += int(any(index >= 0 for index in selected))
        for key, value in increment.items():
            counts[key] = counts.get(key, 0) + value
        sample['reference_bank_indices'] = selected
        sample['metadata']['reference_policy'] = 'fixed_bank_mixture'
        sample['metadata']['actual_reference_sources'] = records
        return sample


def mixed_visual_features(records, vision, bank_features):
    """Encode remaining crops and splice the actual frozen bank rows in order."""
    crops, source_indices = [], []
    for row in records:
        selected = row.get('reference_bank_indices', [-1] * len(row['references']))
        if len(selected) != len(row['references']):
            raise ValueError('Mixed reference layout differs')
        for image, index in zip(row['references'], selected):
            if not -1 <= index < len(bank_features):
                raise ValueError('Invalid reference bank index')
            source_indices.append(index)
            if index < 0:
                crops.append(image)
    features = vision(crops)
    if len(features) != len(crops):
        raise ValueError('Crop encoder omitted a reference')
    result, cursor = [], 0
    for index in source_indices:
        if index < 0:
            result.append(features[cursor])
            cursor += 1
        else:
            result.append(bank_features[index])
    return torch.stack(result) if result else bank_features.new_zeros((0, bank_features.shape[-1]))
