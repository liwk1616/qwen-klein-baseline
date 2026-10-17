"""Causal, text-preserving reference selection for the fixed-bank ablation."""
import json
import re

NAMES = ('fred', 'wilma', 'barney', 'betty', 'pebbles', 'mr slate', 'dino')
PRONOUN = re.compile(r'\b(?:he|she|him|her|his|hers|they|them|their|theirs|himself|herself|themselves)\b', re.I)
SYSTEM = (
    'Resolve person pronouns in the CURRENT Flintstones caption using only the preceding generated images '
    'and supplied captions. The images can contain generation errors; explicit names and current actions '
    'in the captions take precedence. Do not describe or invent a new event. '
    'For each numbered current-caption pronoun, return its named referent(s) from the supplied allowed names. '
    'If uncertain, return an empty names list. Do not assign a person to an inanimate referent. '
    'Return only JSON: {"bindings":[{"index":0,"names":["fred"],"possessive":false}]}. '
    'Use possessive=true only for possessive uses such as his coat or her dress. '
    'Provide exactly one binding for each numbered pronoun. No explanation or additional fields.'
)


def named_people(caption):
    return [name for name in NAMES
            if re.search(r'(?<!\w)' + (r'mr\.?\s+slate' if name == 'mr slate' else re.escape(name)) + r'(?!\w)', caption, re.I)]


def pronouns(caption):
    return [{'index': i, 'text': m.group(), 'start': m.start(), 'end': m.end()}
            for i, m in enumerate(PRONOUN.finditer(caption))]


def caption_sources(request):
    captions = [h['caption'] for h in request['context']['history']] + [request['target_caption']]
    result = {}
    for position, caption in enumerate(captions, 1):
        for name in named_people(caption):
            result.setdefault(name, []).append({'position': position, 'caption': caption})
    return result


def parse_bindings(raw, caption, allowed):
    start = raw.find('{')
    if start < 0:
        raise ValueError('No JSON object in pronoun-resolution response')
    value, _ = json.JSONDecoder().raw_decode(raw[start:])
    if not isinstance(value, dict) or not isinstance(value.get('bindings'), list):
        raise ValueError('Missing bindings list')
    mentions, seen, replacements = pronouns(caption), [], []
    for binding in value['bindings']:
        index, names = binding.get('index'), binding.get('names')
        if type(index) is not int or index not in range(len(mentions)) or index in seen:
            raise ValueError('Invalid or repeated pronoun index')
        seen.append(index)
        if not isinstance(names, list) or any(not isinstance(n, str) for n in names):
            raise ValueError('Invalid referent names')
        names = [n.strip().lower() for n in names]
        if len(names) != len(set(names)) or any(n not in allowed or n not in NAMES for n in names):
            raise ValueError('Referent has no explicit name in the available story captions')
        possessive = binding.get('possessive', False)
        if type(possessive) is not bool:
            raise ValueError('Invalid possessive flag')
        mention = mentions[index]
        word = mention['text'].lower()
        if word in ('he', 'she', 'him', 'her', 'his', 'hers', 'himself', 'herself') and len(names) > 1:
            raise ValueError('A singular pronoun has multiple referents')
        replacement = None
        if names:
            replacement = ' and '.join('Mr Slate' if n == 'mr slate' else n.capitalize() for n in names)
            if possessive or word in ('his', 'hers', 'their', 'theirs'):
                replacement += "'s"
            if word in ('himself', 'herself', 'themselves'):
                replacement += ' ' + word
        replacements.append({**mention, 'names': names, 'replacement': replacement})
    if sorted(seen) != list(range(len(mentions))):
        raise ValueError('Not every current-caption pronoun was resolved or marked uncertain')
    resolved = caption
    for item in sorted(replacements, key=lambda r: r['start'], reverse=True):
        if item['replacement']:
            resolved = resolved[:item['start']] + item['replacement'] + resolved[item['end']:]
    return resolved, sorted(replacements, key=lambda r: r['index'])


def reference_records(request, resolved, bank):
    sources = caption_sources(request)
    names = named_people(resolved)
    if any(name not in sources or name not in bank for name in names):
        raise ValueError('Reference selection escaped the current causal caption prefix')
    return [{'name': name, 'image': bank[name], 'kind': 'official_fixed_character_reference',
             'caption_evidence': sources[name]} for name in names]


def add_reference_delta(story_condition, native_condition, fused_condition):
    """Retain the learned story condition and add the trained identity residual."""
    if story_condition.shape != native_condition.shape or native_condition.shape != fused_condition.shape:
        raise ValueError('Story and character conditions have different token layouts')
    return story_condition + (fused_condition - native_condition)


def combine_reference_condition(story_condition, native_condition, fused_condition, mapper_weight):
    """Scale only the story residual, preserving both established endpoints."""
    if not 0 <= mapper_weight <= 1:
        raise ValueError('Mapper weight must be finite and between zero and one')
    if native_condition.shape != fused_condition.shape:
        raise ValueError('Character conditions have different token layouts')
    if mapper_weight == 0:
        return fused_condition
    if story_condition is None or story_condition.shape != native_condition.shape:
        raise ValueError('A positive mapper weight requires a matching story condition')
    if mapper_weight == 1:
        return add_reference_delta(story_condition, native_condition, fused_condition)
    return native_condition + mapper_weight*(story_condition-native_condition) + (fused_condition-native_condition)
