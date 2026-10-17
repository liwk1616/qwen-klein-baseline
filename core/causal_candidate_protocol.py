"""Causal candidate requests: referring captions, selected history, first-frame bank."""
import json
import re

from independent_candidate_protocol import (SEED_STRIDE, classifier_alignment,
    name_mentions, scene_evidence)
from story_reference_refresh import caption_sources, parse_bindings, pronouns
from storygptv_protocol import causal_history


PROTOCOL = 'klein_coreferential_first_bank_causal_best4_val_v1'


def request_for(story, story_index, position, selected, candidates):
    if story['split'] != 'val' or story['initial_original_images'] or len(story['targets']) != 4:
        raise ValueError('This trial accepts only four-frame validation stories')
    if candidates not in (1, 4):
        raise ValueError('Unsupported candidate budget')
    target = story['targets'][position]
    if position and target['initial_reference_bank']:
        raise ValueError('Only the first frame may receive bank references')
    context = causal_history(story['story_id'], story['targets'], selected, position)
    return {'index': story_index*4+position, 'example_id': target['example_id'],
        'task_id': story['task_id'], 'story_id': story['story_id'], 'clip_id': target['clip_id'],
        'target_position': position+1, 'target_caption': target['caption'], 'context': context,
        'initial_reference_bank': target['initial_reference_bank'] if position == 0 else [],
        'seeds': [int(target['seed'])+SEED_STRIDE*k for k in range(candidates)],
        'protocol': PROTOCOL, 'split': 'val'}


def ranking_target(request, raw=None, error=None):
    """Model-derived identity evidence is only for ranking, never generator rewriting."""
    caption = request['target_caption']
    allowed = caption_sources(request)
    mentions = pronouns(caption)
    resolved, bindings = caption, []
    if mentions and error is None:
        if raw is None:
            error = 'No model resolution available for current pronouns'
        else:
            try:
                resolved, bindings = parse_bindings(raw, caption, allowed)
            except (ValueError, TypeError, KeyError, AttributeError) as problem:
                error = str(problem)
    names = list(dict.fromkeys(item['name'] for item in name_mentions(resolved)))
    if any(name not in allowed for name in names):
        raise ValueError('Ranking identity escaped the allowed caption prefix')
    uncertain = bool(error) or any(not b['names'] for b in bindings) or not names
    return {'named_characters': names, 'scene_evidence': scene_evidence(caption),
        'character_target_uncertain': uncertain, 'bindings': bindings, 'error': error,
        'raw_response': raw, 'caption_name_evidence': allowed,
        'generator_caption': caption, 'inferred_caption_for_ranking_only': resolved,
        'unresolved_pronouns': sum(not b['names'] for b in bindings) if not error else len(mentions)}


def rank_classifier(target, char_probs, bg_probs, bg_names):
    result = classifier_alignment(target, char_probs, bg_probs, bg_names)
    if target['character_target_uncertain']:
        result.update(character=.5, caption_character_exact=None, caption_character_soft=None)
    result['character_target_uncertain'] = target['character_target_uncertain']
    return result


def grader_inputs(request, candidate_image):
    """All allowed context precedes the candidate; the candidate is always last."""
    content, images, evidence = [], [], []
    for item in request['initial_reference_bank']:
        content.extend([{'type': 'text', 'text': 'First-frame identity reference: '+item['name']},
                        {'type': 'image'}])
        images.append(item['image'])
        evidence.append({'kind': 'first_frame_bank', 'name': item['name'], 'image': item['image']})
    for item in request['context']['history']:
        content.extend([{'type': 'text', 'text': 'Previous selected generated frame; caption: '+item['caption']},
                        {'type': 'image'}])
        images.append(item['image']); evidence.append(dict(item))
    content.extend([{'type': 'text', 'text': 'Candidate current frame (LAST image):'}, {'type': 'image'},
                    {'type': 'text', 'text': 'Current referring caption: '+json.dumps(request['target_caption'])}])
    images.append(candidate_image)
    return content, images, evidence


def parse_grade_response(raw):
    """Accept JSON or an unambiguous six-line field response, never partial grades."""
    from independent_candidate_protocol import parse_visual_scores, VLM_FIELDS
    start = raw.find('{')
    if start >= 0:
        value, _ = json.JSONDecoder().raw_decode(raw[start:])
        format_name = 'json'
    else:
        value = {}
        for line in raw.strip().splitlines():
            match = re.fullmatch(r'(visible|identity|cast|background|action|quality):\s*(.*)', line.strip())
            if not match or match[1] in value:
                raise ValueError('Unexpected or duplicate labeled grading field')
            key, field = match.groups()
            if key in VLM_FIELDS:
                if not re.fullmatch(r'[0-4]', field):
                    raise ValueError('Labeled grades must be integers from zero to four')
                value[key] = int(field)
            else:
                value[key] = field
        format_name = 'labeled_lines'
    if not isinstance(value, dict) or set(value) != {'visible', *VLM_FIELDS}:
        raise ValueError('Incomplete or unexpected grading schema')
    value = dict(value)
    visible = value.pop('visible')
    if not isinstance(visible, str) or not visible.strip():
        raise ValueError('No visible-image description')
    return visible, parse_visual_scores(json.dumps(value)), format_name


def select_local(request, candidates):
    from independent_candidate_protocol import candidate_rank
    if sorted(c['candidate_index'] for c in candidates) != list(range(len(request['seeds']))):
        raise ValueError('Incomplete or duplicate candidate pool')
    ranked = []
    for candidate in candidates:
        k = candidate['candidate_index']
        if (candidate['seed'] != request['seeds'][k]
                or candidate['base_example_id'] != request['example_id']
                or candidate['generated_history'] != request['context']['history']):
            raise ValueError('Candidate seed, identity, or selected history differs')
        score, components = candidate_rank(candidate['classifier'], candidate['visual'])
        ranked.append({'candidate_index': k, 'score': score, 'components': components})
    chosen = max(ranked, key=lambda c: (c['score'], -c['candidate_index']))
    return chosen['candidate_index'], ranked
