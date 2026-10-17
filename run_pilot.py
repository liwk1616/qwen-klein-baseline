"""Replay the frozen twenty-case short/stride-8 study with shared first frames."""
import argparse
import json
from pathlib import Path
import pickle
import os

from runtime import (ROOT, asset_settings, metrics_config, gpu_ids, read,
                     mappings, workers, restore_plan)
from archive_io import atomic_json as write
from pilot_frame import make_request, generate_frame, PROTOCOL

ARMS = ('short_contextual', 'long_stride8', 'short_storygptv')


def validate_cases(cases):
    seen = set()
    for index, case in enumerate(cases):
        key = case['case_id']
        if not key.isdigit() or key in seen or case['case_index'] != index:
            raise ValueError('Expected unique numeric case IDs and original contiguous indices')
        seen.add(key)
        first = case['arms'][ARMS[0]][0]
        for arm, count in zip(ARMS, (5, 5, 4)):
            targets = case['arms'][arm]
            if len(targets) != count or targets[0] != first:
                raise ValueError('Pilot arms must have 5/5/4 targets and the same first frame')
            if first['clip_id'] != case['story_id']:
                raise ValueError('First clip and story ID differ')
            for position, target in enumerate(targets):
                if (target['position'] != position or target['split'] != 'test'
                        or not target['caption'].strip() or type(target['seed']) is not int):
                    raise ValueError('Invalid pilot caption, position, split or seed')
                if position and target['initial_reference_bank']:
                    raise ValueError('Unexpected initial references on a continuation frame')
    if not cases:
        raise ValueError('No cases')


def prepare(args):
    config = metrics_config()
    settings = asset_settings()
    cases = read(args.cases or settings['pilot_cases'])
    validate_cases(cases)
    if args.limit_cases is not None:
        if args.limit_cases < 1:
            raise ValueError('--limit-cases must be positive')
        cases = cases[:args.limit_cases]
    gpus = gpu_ids(args.gpus)
    if len(gpus) > len(cases):
        raise ValueError('Use no more GPUs than cases')
    from transformers import AutoTokenizer
    from prepare_char_klein import caption_tokens
    from klein_style_condition import klein_prompt
    tokenizer = AutoTokenizer.from_pretrained(Path(config['model'])/'tokenizer', local_files_only=True)
    max_tokens = max(len(caption_tokens(tokenizer, klein_prompt(t['caption']))[1]['input_ids'])
                     for c in cases for arm in ARMS for t in c['arms'][arm])
    if max_tokens > 128:
        raise ValueError('Pilot caption exceeds the frozen mapper budget')
    bank = {r['name']: r['path'] for r in read(Path(config['data'])/'reference_bank.json')}
    for case in cases:
        for arm in ARMS:
            for reference in case['arms'][arm][0]['initial_reference_bank']:
                if reference['image'] != bank[reference['name']]:
                    raise ValueError('Pilot and configured fixed bank differ')
    summary = {'cases': len(cases), 'logical_frames': len(cases)*14,
               'unique_generated_frames': len(cases)*12, 'candidates': len(cases)*48,
               'shared_first_frame': True, 'max_caption_tokens': max_tokens}
    if args.check_only:
        print(json.dumps(summary))
        return
    if args.output is None:
        raise ValueError('--output is required')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    plan = {**config, **summary, 'protocol': PROTOCOL, 'variant': 'history_pilot20',
            'gpus': gpus, 'path_mappings': mappings(), 'python': settings['python_qwen'],
            'score_python': settings['python_flux'], 'candidates_per_frame': 4,
            'new_training_steps': 0, 'formal_contract_benchmark': False,
            'original_target_images_used': False, 'future_captions_used': False,
            'evaluation_labels_used_for_candidate_selection': False}
    write(root/'run_plan.json', plan)
    write(root/'inputs/cases.json', cases)
    print(json.dumps({**summary, 'output': str(root)}))


def worker(root, shard):
    import models as baseline
    from progressive_protocol import target_v1b, classify_v1b
    plan = restore_plan(root)
    if not 0 <= shard < len(plan['gpus']) or os.environ.get('CUDA_VISIBLE_DEVICES') != str(plan['gpus'][shard]):
        raise ValueError('Wrong worker GPU binding')
    baseline.ranking_target, baseline.rank_classifier = target_v1b, classify_v1b
    models = baseline.Models(plan)
    cases = [c for c in read(root/'inputs/cases.json') if c['case_index'] % len(plan['gpus']) == shard]
    completed = 0
    with models.torch.inference_mode():
        for case in cases:
            first, _ = generate_frame(models, root, make_request(case, ARMS[0], 0, []))
            completed += 1
            for arm in ARMS:
                selected = [first]
                for position in range(1, len(case['arms'][arm])):
                    output, _ = generate_frame(models, root, make_request(case, arm, position, selected))
                    selected.append(output)
                    completed += 1
                write(root/'frames'/f"case{case['case_id']}_{arm}_sequence.json", selected)
                write(root/f'worker_{shard:03d}.json', {'status': 'running', 'frames': completed})
    write(root/f'worker_{shard:03d}.json', {'status': 'complete', 'frames': completed})


def finish(root):
    from causal_candidate_protocol import select_local
    from run_metrics import summarize
    from runtime_io import write_rows
    plan = restore_plan(root)
    with (Path(plan['labels'])/'labels.pkl').open('rb') as stream:
        chars = pickle.load(stream)
    with (Path(plan['labels'])/'labels_bg.pkl').open('rb') as stream:
        backgrounds = pickle.load(stream)['label']
    values, unique = [], set()
    for case in read(root/'inputs/cases.json'):
        for arm in ARMS:
            sequence = read(root/'frames'/f"case{case['case_id']}_{arm}_sequence.json")
            if len(sequence) != len(case['arms'][arm]):
                raise ValueError('Incomplete pilot sequence')
            history = []
            for position, output in enumerate(sequence):
                request = make_request(case, arm, position, history)
                directory = root/'frames'/request['example_id']
                if read(directory/'request.json') != request:
                    raise ValueError('Saved request has a different causal prefix')
                pool = read(directory/'candidates.json')
                choice, ranking = select_local(request, pool)
                if read(directory/'selection.json') != {'choice': choice, 'ranking': ranking}:
                    raise ValueError('Saved selection does not match the frozen ranking rule')
                if output['image'] != pool[choice]['image'] or not Path(output['image']).is_file():
                    raise ValueError('Selected image missing or inconsistent')
                row = {'case_id': case['case_id'], 'arm': arm, 'position': position+1,
                       'clip_id': request['clip_id'], 'image': output['image']}
                for mode, labels, field in [('char', chars, 'character'), ('bg', backgrounds, 'background')]:
                    row[mode+'_target_indices'] = [i for i,v in enumerate(labels[request['clip_id']]) if v == 1]
                    row[mode+'_predicted_indices'] = [i for i,p in enumerate(pool[choice]['classifier'][field+'_probabilities']) if p > .5]
                values.append(row)
                history.append(output)
                unique.add(request['example_id'])
    if len(values) != plan['logical_frames'] or len(unique) != plan['unique_generated_frames']:
        raise ValueError('Pilot coverage differs from the prepared plan')
    results = {}
    for arm in ARMS:
        selected = [r for r in values if r['arm'] == arm]
        results[arm] = {scope: {mode: summarize(rows, mode) for mode in ('char', 'bg')}
                        for scope, rows in [('all_positions', selected),
                                            ('continuation_only', [r for r in selected if r['position'] > 1])]}
    write_rows(root/'predictions.jsonl', values)
    write(root/'result.json', {'status': 'complete', 'metrics': results, 'logical_frames': len(values),
                              'unique_generated_frames': len(unique), 'formal_contract_benchmark': False})
    print(json.dumps(results, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'worker', 'finish'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cases', type=Path)
    parser.add_argument('--limit-cases', type=int)
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare':
        return prepare(args)
    if args.output is None:
        parser.error('--output is required')
    root = args.output.resolve()
    if args.action == 'run':
        plan = restore_plan(root)
        workers([(gpu, [plan['python'], '-B', __file__, 'worker', '--output', root, '--shard', shard])
                 for shard, gpu in enumerate(plan['gpus'])])
        finish(root)
    elif args.action == 'worker':
        worker(root, args.shard)
    else:
        finish(root)


if __name__ == '__main__':
    main()
