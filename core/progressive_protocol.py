"""Versioned, conservative resolver handling for the fixed validation cohort."""
from copy import deepcopy
import json

from causal_candidate_protocol import ranking_target as legacy_target, rank_classifier as legacy_classifier
from independent_candidate_protocol import CLASSIFIER_NAMES, name_mentions, scene_evidence
from story_reference_refresh import caption_sources, parse_bindings, pronouns, NAMES


def recover_json_container(raw):
    """Close at most two containers, only after an already closed inner value.

    Never add a quote, scalar, name, field, comma or incomplete list item.
    The caller must still validate every expected binding and its semantics.
    """
    if not isinstance(raw, str):
        return raw, False
    start = raw.find('{')
    if start < 0:
        return raw, False
    text = raw[start:].strip()
    if text.endswith('```'):
        text = text[:-3].rstrip()
    try:
        json.JSONDecoder().raw_decode(text)
        return raw, False
    except ValueError:
        pass
    if not text or text[-1] not in '}]':
        return raw, False
    stack, quoted, escaped = [], False, False
    for character in text:
        if quoted:
            if escaped:
                escaped = False
            elif character == '\\':
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character in '{[':
            stack.append(character)
        elif character in '}]':
            if not stack or stack.pop() != ('{' if character == '}' else '['):
                return raw, False
    if quoted or escaped or not 1 <= len(stack) <= 2:
        return raw, False
    repaired = text+''.join('}' if c == '{' else ']' for c in reversed(stack))
    try:
        value = json.loads(repaired)
    except ValueError:
        return raw, False
    if not isinstance(value, dict) or not isinstance(value.get('bindings'), list):
        return raw, False
    return repaired, True


def target_v1a(request, raw=None, error=None):
    repaired, changed = recover_json_container(raw)
    target = legacy_target(request, repaired, error)
    target.update(raw_response=raw, parsed_response=repaired, format_recovered=changed,
                  parser_version='V1a_container_recovery')
    return target


def target_v1b(request, raw=None, error=None):
    repaired, changed = recover_json_container(raw)
    caption, mentions = request['target_caption'], pronouns(request['target_caption'])
    allowed = caption_sources(request)
    diagnostics, records, accepted = [], [], []
    values = []
    parse_error = error
    if mentions and parse_error is None:
        try:
            if not isinstance(repaired, str) or '{' not in repaired:
                raise ValueError('No resolver response object')
            value, _ = json.JSONDecoder().raw_decode(repaired[repaired.find('{'):])
            if not isinstance(value, dict) or not isinstance(value.get('bindings'), list):
                raise ValueError('Missing bindings list')
            values = value['bindings']
        except (ValueError, TypeError) as problem:
            parse_error = str(problem)
    resolved = caption
    for mention in mentions:
        i = mention['index']
        matches = [b for b in values if isinstance(b, dict) and type(b.get('index')) is int and b['index'] == i]
        problems, known, replacement = [], [], None
        if len(matches) != 1:
            problems.append('Missing or duplicate binding')
        else:
            binding = matches[0]
            names = binding.get('names')
            if not isinstance(names, list) or any(not isinstance(n, str) for n in names):
                problems.append('Invalid names array')
            else:
                normalized = [n.strip().lower() for n in names]
                known = list(dict.fromkeys(n for n in normalized if n in allowed and n in NAMES))
                if len(normalized) != len(set(normalized)):
                    problems.append('Duplicate names')
                if any(n not in allowed or n not in NAMES for n in normalized):
                    problems.append('Name or unnamed entity outside the supported named schema')
                if mention['text'].lower() in ('he', 'she', 'him', 'her', 'his', 'hers', 'himself', 'herself') and len(normalized) > 1:
                    # Do not choose one actor from an invalid singular group.
                    known = []; problems.append('Singular pronoun has multiple referents')
                if not isinstance(binding.get('possessive', False), bool):
                    known = []; problems.append('Invalid possessive flag')
                if known and not problems:
                    # Reuse the established text-span replacement, including possessives.
                    one_caption = caption[mention['start']:mention['end']]
                    replacement, _ = parse_bindings(json.dumps({'bindings': [
                        {'index': 0, 'names': known, 'possessive': binding.get('possessive', False)}]}), one_caption, allowed)
                    # Reflexive forms intentionally still contain the reflexive pronoun.
        complete = bool(known) and not problems
        records.append({**mention, 'names': known, 'replacement': replacement if complete else None,
                        'status': 'resolved_named' if complete else 'unresolved_or_unnamed', 'problems': problems})
        accepted.extend(known)
        diagnostics.extend({'index': i, 'problem': p} for p in problems)
    for binding in reversed(records):
        if binding['replacement'] is not None:
            resolved = resolved[:binding['start']]+binding['replacement']+resolved[binding['end']:]
    explicit = list(dict.fromkeys(m['name'] for m in name_mentions(caption)))
    known = list(dict.fromkeys(explicit+accepted))
    complete = bool(known) and parse_error is None and all(b['status'] == 'resolved_named' for b in records)
    scope = 'full_named_set' if complete else ('partial_known' if known else 'unknown')
    return {'named_characters': known, 'explicit_current_names': explicit,
        'scene_evidence': scene_evidence(caption), 'character_target_uncertain': not complete,
        'character_scope': scope, 'bindings': records, 'binding_diagnostics': diagnostics,
        'error': parse_error, 'raw_response': raw, 'parsed_response': repaired,
        'format_recovered': changed, 'caption_name_evidence': allowed,
        'generator_caption': caption, 'inferred_caption_for_ranking_only': resolved,
        'resolved_caption_for_generation': resolved,
        'unresolved_pronouns': sum(b['status'] != 'resolved_named' for b in records),
        'parser_version': 'V1b_partial_known',
        'unknown_entity_policy': 'Unresolved and unnamed entities are not forced into seven named identities'}


def classify_v1b(target, char_probs, bg_probs, bg_names):
    score = legacy_classifier(target, char_probs, bg_probs, bg_names)
    scope = target['character_scope']
    if scope == 'partial_known':
        indices = [CLASSIFIER_NAMES.index(name) for name in target['named_characters']]
        presence = sum(char_probs[i] for i in indices)/len(indices)
        score.update(character=presence, caption_character_exact=None, caption_character_soft=presence)
    score['character_scope'] = scope
    return score


def generator_request(request, target, rewrite):
    result = deepcopy(request)
    if rewrite:
        result['target_caption'] = target['resolved_caption_for_generation']
        result['context']['current_caption'] = result['target_caption']
    result['raw_target_caption'] = request['target_caption']
    result['caption_rewritten'] = result['target_caption'] != request['target_caption']
    return result
