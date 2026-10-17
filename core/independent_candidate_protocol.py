"""Independent caption inputs and label-free candidate selection rules."""
import json
import math
import re

from klein_style_condition import klein_prompt
from prepare_char_klein import caption_tokens
from story_reference_refresh import NAMES


PROTOCOL = 'd1_independent_caption_candidates_v1'
CLASSIFIER_NAMES = ('wilma', 'fred', 'betty', 'barney', 'dino', 'pebbles', 'mr slate')
SEED_STRIDE = 1_000_003
RANK_WEIGHTS = {'character': .45, 'background': .25, 'visual': .30}
VLM_FIELDS = ('identity', 'cast', 'background', 'action', 'quality')
SCENE_GROUPS = {
    'living room': ['living room', 'livingroom'], 'kitchen': ['kitchen'],
    'bedroom': ['bedroom'], 'bathroom': ['bathroom'], 'dining room': ['dining room'],
    'office': ['office', 'inside office'], 'quarry office': ['quarry office'],
    'outside': ['outside', 'outdoors', 'outdoor'], 'room': ['room', 'in room'],
    'road': ['road', 'street', 'roadway'], 'yard': ['yard', 'backyard', 'back yard', 'front yard'],
    'beach': ['beach'], 'quarry': ['quarry'], 'barn': ['barn'], 'park': ['park'],
    'garden': ['garden'], 'garage': ['garage'], 'doorway': ['doorway', 'front doorway', 'front door'],
    'porch': ['porch', 'front porch'], 'bowling alley': ['bowling alley', 'bowling ally'],
    'restaurant': ['restaurant'], 'diner': ['diner'], 'cafe': ['cafe'],
    'store': ['store'], 'school': ['school'], 'classroom': ['classroom'],
    'hospital': ['hospital'], 'courtroom': ['courtroom', 'court room'],
    'laboratory': ['laboratory', 'lab'], 'theater': ['theater', 'theatre'],
    'pool': ['pool', 'swimming pool'], 'forest': ['forest'], 'desert': ['desert'],
    'mountain': ['mountain', 'mountains'], 'cave': ['cave'], 'lake': ['lake'],
    'river': ['river'], 'sea': ['sea', 'ocean'], 'hallway': ['hallway'],
    'airport': ['airport'], 'hotel': ['hotel'], 'hotel lobby': ['hotel lobby'],
    'stadium': ['stadium'], 'jail': ['jail', 'jail cell'], 'farm': ['farm'],
}
SPECIFIC_INTERIORS = ('living room', 'kitchen', 'bedroom', 'bathroom', 'dining room',
                      'office', 'quarry office', 'garage', 'classroom', 'courtroom', 'laboratory')


def name_mentions(caption):
    result = []
    for name in NAMES:
        pattern = r'mr\.?\s+slate' if name == 'mr slate' else re.escape(name)
        for match in re.finditer(r'(?<!\w)'+pattern+r'(?!\w)', caption, re.I):
            result.append({'name': name, 'start': match.start(), 'end': match.end(), 'text': match.group()})
    return sorted(result, key=lambda r: r['start'])


def scene_evidence(caption):
    matches = []
    for group, aliases in SCENE_GROUPS.items():
        for phrase in aliases:
            for match in re.finditer(r'(?<!\w)'+re.escape(phrase)+r'(?!\w)', caption, re.I):
                matches.append({'group': group, 'start': match.start(), 'end': match.end(), 'text': match.group()})
    # Keep specific phrases such as "living room", not their nested "room".
    kept = []
    for item in sorted(matches, key=lambda r: (-(r['end']-r['start']), r['start'])):
        if not any(item['start'] < r['end'] and item['end'] > r['start'] for r in kept):
            kept.append(item)
    return sorted(kept, key=lambda r: r['start'])


def generation_prompt(caption, variant):
    if variant not in ('raw', 'scene'):
        raise ValueError('Unknown prompt variant')
    evidence = scene_evidence(caption)
    text = caption
    if variant == 'scene' and evidence:
        phrases = list(dict.fromkeys(r['text'] for r in evidence))
        text += '\nSetting focus: '+', '.join(phrases)+'.'
        if re.search(r'close[ -]?up|close view|only (?:his|her|the) (?:face|head)', caption, re.I):
            text += ' Keep the stated setting consistent with this framing.'
        else:
            text += ' Keep the described setting visible around the characters while preserving the requested action.'
    return klein_prompt(text)


def bind_request(tokenizer, request):
    caption, prompt = request['target_caption'], request['prompt']
    if prompt != generation_prompt(caption, request['variant']):
        raise ValueError('Generation prompt differs from the recorded source caption')
    _, encoded, _ = caption_tokens(tokenizer, prompt)
    length = max(128, 128*((len(encoded['input_ids'])+127)//128))
    if length > 512:
        raise ValueError('Refusing native caption truncation')
    _, encoded, start = caption_tokens(tokenizer, prompt, length)
    bound = []
    for mention in name_mentions(caption):
        a, b = start+mention['start'], start+mention['end']
        tokens = [i for i, (x, y) in enumerate(encoded['offset_mapping']) if y > x and x < b and y > a]
        if not tokens:
            raise ValueError('A named reference has no native token')
        bound.append({**mention, 'token_index': tokens[-1]})
    if len({r['token_index'] for r in bound}) != len(bound):
        raise ValueError('Reference bindings collide')
    return length, bound


def make_request(story_index, story, position, original_caption, bank, variant, candidates):
    target = story['targets'][position]
    caption = original_caption.strip()
    names = [name for name in NAMES if any(m['name'] == name for m in name_mentions(caption))]
    if candidates not in (1, 4):
        raise ValueError('Only the declared single or four-candidate budget is supported')
    return {'index': story_index*4+position, 'example_id': target['example_id'],
            'story_index': story_index, 'story_id': story['story_id'], 'task_id': story['task_id'],
            'clip_id': target['clip_id'], 'target_position': position+1, 'split': story['split'],
            'protocol': PROTOCOL, 'variant': variant, 'target_caption': caption,
            'released_story_caption': target['caption'], 'prompt': generation_prompt(caption, variant),
            'named_characters': names, 'scene_evidence': scene_evidence(caption),
            'references': [{'name': name, 'image': bank[name]} for name in names],
            'seeds': [int(target['seed'])+SEED_STRIDE*k for k in range(candidates)],
            'context': {'history': []}, 'original_target_images_used': False}


def parse_visual_scores(raw):
    start = raw.find('{')
    if start < 0:
        raise ValueError('Visual grader returned no JSON object')
    value, _ = json.JSONDecoder().raw_decode(raw[start:])
    if not isinstance(value, dict) or set(value) != set(VLM_FIELDS):
        raise ValueError('Visual grader returned an unexpected schema')
    if any(type(value[k]) is not int or not 0 <= value[k] <= 4 for k in VLM_FIELDS):
        raise ValueError('Visual grades must be integers from zero to four')
    return value


def classifier_alignment(request, char_probabilities, bg_probabilities, bg_names):
    if len(char_probabilities) != 7 or len(bg_probabilities) != len(bg_names):
        raise ValueError('Unexpected classifier dimensions')
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in char_probabilities+bg_probabilities):
        raise ValueError('Invalid classifier probabilities')
    # These targets come only from caption names, never evaluation label files.
    desired = [name in request['named_characters'] for name in CLASSIFIER_NAMES]
    exact = all((p > .5) == y for p, y in zip(char_probabilities, desired))
    soft = sum(p if y else 1-p for p, y in zip(char_probabilities, desired))/7
    character = .5*float(exact)+.5*soft
    groups = list(dict.fromkeys(r['group'] for r in request['scene_evidence']))
    # Generic "room" is too ambiguous to define a specific background target.
    requested = [g for g in groups if g != 'room']
    probabilities = dict(zip(bg_names, bg_probabilities))
    positives = [max(probabilities.get(alias, 0.) for alias in SCENE_GROUPS[g]) for g in requested]
    if positives:
        background = sum(positives)/len(positives)
        if any(g in SPECIFIC_INTERIORS for g in requested):
            competing = [probabilities.get(alias, 0.) for g in SPECIFIC_INTERIORS if g not in groups
                         for alias in SCENE_GROUPS[g]]
            background = .8*background+.2*(1-max(competing, default=0.))
    else:
        background = .5
    return {'character': character, 'background': background,
            'caption_character_exact': exact, 'caption_character_soft': soft,
            'background_groups_from_caption': requested}


def candidate_rank(classifier, visual):
    error = visual.get('error')
    ratings = visual.get('scores')
    if error is None and (ratings is None or parse_visual_scores(json.dumps(ratings)) != ratings):
        raise ValueError('Malformed visual scores')
    # A grading failure preserves every candidate and receives a neutral grade.
    fidelity = sum(ratings[k] for k in VLM_FIELDS)/20 if error is None else .5
    values = {k: classifier[k] for k in ('character', 'background')}
    values['visual'] = fidelity
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
        raise ValueError('Candidate ranking inputs must be bounded')
    return sum(RANK_WEIGHTS[k]*v for k, v in values.items()), values


def select_candidate(request, candidates, classifier, visual):
    expected = list(range(len(request['seeds'])))
    if sorted(r['candidate_index'] for r in candidates) != expected:
        raise ValueError('Missing or repeated candidate; cannot select from an incomplete pool')
    ranking = []
    for value in candidates:
        k, key = value['candidate_index'], value['example_id']
        if (value['base_example_id'] != request['example_id'] or value['frame_index'] != request['index']
                or value['seed'] != request['seeds'][k] or value['status'] != 'complete'):
            raise ValueError('Candidate belongs to a different request or seed')
        score, components = candidate_rank(classifier[key], visual[key])
        ranking.append({'candidate_index': k, 'candidate_example_id': key, 'score': score,
                        'components': components, 'visual_grading_error': visual[key].get('error')})
    chosen = max(ranking, key=lambda r: (r['score'], -r['candidate_index']))
    return chosen['candidate_index'], sorted(ranking, key=lambda r: r['candidate_index'])
