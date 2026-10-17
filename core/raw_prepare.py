# Extracted from scripts/flintstones_klein/run_independent_candidate_test.py; historical orchestration removed.
import json
from asset_paths import loads as asset_loads
from collections import Counter
from archive_io import atomic_json
from runtime_io import rows, write_rows
from independent_candidate_protocol import PROTOCOL, make_request, bind_request, select_candidate

def pool_prepare(root, common, stories, raw, bank, taxonomy, variant, candidates, tokenizer):
    root.mkdir(parents=True, exist_ok=False)
    (root/'inputs').mkdir()
    requests, manifest, candidate_manifest = [], [], []
    lengths, scene_changed = Counter(), 0
    for i, story in enumerate(stories):
        for j, target in enumerate(story['targets']):
            request = make_request(i, story, j, raw[target['clip_id']]['caption'], bank, variant, candidates)
            length, bindings = bind_request(tokenizer, request)
            request.update(native_sequence_length=length, bindings=bindings)
            lengths[length] += 1
            scene_changed += variant == 'scene' and bool(request['scene_evidence'])
            requests.append(request)
            item = {key: request[key] for key in ['index', 'example_id', 'task_id', 'story_id', 'clip_id', 'target_position']}
            manifest.append(item)
            for k in range(candidates):
                candidate_manifest.append({**item, 'index': request['index']*candidates+k,
                                           'example_id': request['example_id']+f'__candidate_{k:02d}'})
    split = stories[0]['split']
    if any(s['split'] != split for s in stories):
        raise ValueError('A pool must have exactly one held-out split')
    plan = {**common, 'protocol': PROTOCOL, 'split': split, 'stories': len(stories),
            'frames': len(requests), 'variant': variant, 'candidates_per_frame': candidates,
            'planned_candidates': len(candidate_manifest), 'scene_prompt_changed_frames': scene_changed}
    atomic_json(root/'run_plan.json', plan)
    atomic_json(root/'authorization.json', {'independent_original_captions_allowed': True,
        'fixed_reference_bank_every_frame_allowed': True, 'caption_based_candidate_selection_allowed': True,
        'gpus': common['gpus'], 'scope': common['user_authorization']})
    write_rows(root/'inputs/stories.jsonl', stories)
    write_rows(root/'inputs/requests.jsonl', requests)
    write_rows(root/'inputs/evaluation.jsonl', manifest)
    write_rows(root/'inputs/candidate_evaluation.jsonl', candidate_manifest)
    atomic_json(root/'inputs/background_names.json', taxonomy)
    atomic_json(root/'inputs/reference_bank.json', [{'name': k, 'path': v} for k, v in bank.items()])
    atomic_json(root/'preflight.json', {'status': 'passed', 'frames': len(requests),
        'candidates': len(candidate_manifest), 'native_sequence_lengths': dict(lengths),
        'frames_with_references': sum(bool(r['references']) for r in requests),
        'all_named_mentions_have_native_bindings': True, 'no_generation_history': True,
        'reference_source': 'Fixed bank, selected only by names in the current original caption',
        'scene_prompt_changed_frames': scene_changed, 'labels_used_for_selection': False})
    return plan


def make_views(root, output):
    requests = rows(root/'inputs/requests.jsonl')
    plan = asset_loads((root/'run_plan.json').read_text())
    single, selected, decisions = [], [], []
    if plan['candidates_per_frame'] > 1:
        classifier = {r['example_id']: r for r in rows(root/'ranker/classifier/scores.jsonl')}
        visual_rows = [r for p in sorted((root/'ranker/visual').glob('rank_*/scores.jsonl')) for r in rows(p)]
        visual = {r['example_id']: r for r in visual_rows}
        ids = {r['example_id'] for r in output}
        if len(visual_rows) != len(ids) or set(visual) != ids or set(classifier) != ids:
            raise ValueError('Candidate grades do not cover every generated candidate exactly once')
    else:
        visual = {}
    def selected_record(value, request, mode):
        return {**value, 'index': request['index'], 'example_id': request['example_id'],
                'candidate_example_id': value['example_id'], 'selection_mode': mode,
                'kind': 'independent_story_frame', 'candidate_selection_uses_evaluation_labels': False}
    for request in requests:
        group = output[request['index']*plan['candidates_per_frame']:(request['index']+1)*plan['candidates_per_frame']]
        single.append(selected_record(group[0], request, 'single'))
        if plan['candidates_per_frame'] > 1:
            chosen, ranks = select_candidate(request, group, classifier, visual)
            selected.append(selected_record(group[chosen], request, 'best4'))
            decisions.append({'index': request['index'], 'example_id': request['example_id'],
                              'selected_candidate': chosen, 'ranking': ranks})
    views = {'single': single}
    if selected: views['best4'] = selected
    for name, records in views.items():
        path = root/'views'/name/'generated/rank_000'
        path.mkdir(parents=True, exist_ok=False)
        write_rows(path/'generated.jsonl', records)
    write_rows(root/'selection.jsonl', decisions)
    atomic_json(root/'selection_audit.json', {'status': 'passed', 'frames': len(requests),
        'candidates_per_frame': plan['candidates_per_frame'], 'all_candidates_retained': True,
        'selected_candidate_counts': dict(Counter(r['selected_candidate'] for r in decisions)),
        'visual_grading_errors': sum(r.get('error') is not None for r in visual.values()),
        'visual_graded_candidates': len(visual), 'evaluation_labels_used': False,
        'all_single_outputs_use_candidate_zero': True})
    return views
