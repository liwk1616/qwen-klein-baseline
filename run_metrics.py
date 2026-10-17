"""Fresh Flintstones runs using the retained inference and ranking protocols."""
import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import time

from runtime import (ROOT, gpu_ids, read, workers, asset_settings, metrics_config, restore_plan, mappings)

sys.path.insert(0, str(ROOT / 'core'))
from archive_io import atomic_json
from runtime_io import rows, write_rows
from causal_candidate_protocol import request_for, select_local
from progressive_protocol import target_v1b, classify_v1b

VARIANTS = ('raw_single', 'raw_best4', 'history_best4')


def configured_stories(config, variant, override=None, limit=None):
    raw = variant.startswith('raw_')
    source = Path(override or config['raw_stories' if raw else 'stories'])
    stories = rows(source)
    if not stories or (limit is not None and limit < 1):
        raise ValueError('Expected at least one complete story')
    # Preserve the original full-test seed rule, including the fixed 87-story subset.
    if raw:
        subset = {s['story_id']: i for i, s in enumerate(rows(config['raw_seed_subset']))}
        for index, story in enumerate(stories):
            for position, target in enumerate(story['targets']):
                target['seed'] = 20260916 + subset.get(story['story_id'], index) * 4 + position
    if limit is not None:
        stories = stories[:limit]
    seen = set()
    for story in stories:
        if len(story['targets']) != 4 or story['initial_original_images']:
            raise ValueError('Expected four targets and no original target images in generation input')
        if story['split'] != ('test' if raw else 'val'):
            raise ValueError('raw uses the released test split; history_best4 uses development val')
        for position, target in enumerate(story['targets']):
            key = target['example_id']
            if key in seen or not key or Path(key).name != key or key in ('.','..'):
                raise ValueError('Invalid or repeated example_id')
            seen.add(key)
            if not target['caption'].strip() or type(target['seed']) is not int:
                raise ValueError('Missing caption or explicit integer seed')
            if position and target['initial_reference_bank']:
                raise ValueError('Released causal inputs may supply bank images only at frame one')
    return source.resolve(), stories


def prepare(args):
    config = metrics_config(args.config)
    source, stories = configured_stories(config, args.variant, args.stories, args.limit_stories)
    gpus = gpu_ids(args.gpus)
    if len(gpus) > len(stories):
        raise ValueError('Use no more GPUs than stories')
    candidates = 1 if args.variant.endswith('single') else 4
    summary = {'variant': args.variant, 'stories': len(stories), 'frames': len(stories)*4,
               'candidates_per_frame': candidates, 'source': str(source), 'gpus': gpus, 'path_mappings': mappings()}
    if args.check_only:
        print(json.dumps(summary))
        return
    if args.output is None:
        raise ValueError('--output is required for a fresh run')
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError('Use a new output directory: ' + str(root))
    paths = asset_settings()
    plan = {**config, **summary, 'source_config': str(args.config.resolve()),
        'split': stories[0]['split'], 'development_validation': not args.variant.startswith('raw_'),
        'changed_reference_protocol': args.variant == 'history_best4', 'original_target_images_used': False,
        'evaluation_labels_used_for_candidate_selection': False,
        'official_classifiers_used_in_candidate_selection': candidates > 1,
        'python': paths['python_qwen'], 'score_python': paths['python_flux'],
        'user_authorization': 'Explicit invocation of this new research run',
        'prepared_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    if args.variant.startswith('raw_'):
        from transformers import AutoTokenizer
        from raw_prepare import pool_prepare
        plan.update(character_checkpoint=config['raw_character_checkpoint'], mapper_weight=0.,
                    reference_delta_weight=1., protocol='d1_independent_caption_candidates_v1')
        annotations = read(config['raw_annotations'])
        bank = {r['name']: r['path'] for r in read(Path(config['data']) / 'reference_bank.json')}
        tokenizer = AutoTokenizer.from_pretrained(Path(config['model']) / 'tokenizer', local_files_only=True)
        pool_prepare(root, plan, stories, annotations, bank, read(config['background_names']),
                     'raw', candidates, tokenizer)
        saved = read(root / 'run_plan.json')
        saved['research_variant'] = args.variant
        atomic_json(root / 'run_plan.json', saved)
    else:
        root.mkdir(parents=True, exist_ok=False)
        plan.update(protocol='automatic_named_bank_own_history_20261004_v1')
        atomic_json(root / 'run_plan.json', plan)
        write_rows(root / 'inputs/stories.jsonl', stories)
        manifest = []
        for story in stories:
            for position, target in enumerate(story['targets'], 1):
                manifest.append({'index': len(manifest), 'example_id': target['example_id'],
                    'task_id': story['task_id'], 'story_id': story['story_id'],
                    'clip_id': target['clip_id'], 'target_position': position})
        write_rows(root / 'inputs/evaluation.jsonl', manifest)
    print(json.dumps({**summary, 'output': str(root)}, ensure_ascii=False))


def causal_worker(root, shard):
    import models as baseline
    from current_bank import automatic_request, condition
    plan = restore_plan(root)
    gpus = plan['gpus']
    if not 0 <= shard < len(gpus) or os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpus[shard]):
        raise ValueError('Wrong worker GPU binding')
    if plan['variant'] == 'history_best4':
        baseline.ranking_target, baseline.rank_classifier = target_v1b, classify_v1b
    model = baseline.Models(plan)
    torch = model.torch
    selected_stories = [(i,s) for i,s in enumerate(rows(root/'inputs/stories.jsonl')) if i % len(gpus) == shard]
    started, completed = time.monotonic(), 0
    status_path = root / f'worker_{shard:03d}.json'
    with torch.inference_mode():
        for story_index, story in selected_stories:
            history = []
            for position in range(4):
                request = request_for(story, story_index, position, history, plan['candidates_per_frame'])
                folder = root / 'frames' / request['example_id']
                folder.mkdir(parents=True, exist_ok=False)
                atomic_json(folder / 'request.json', request)
                target = model.resolve_for_ranking(request)
                atomic_json(folder / 'ranking_target.json', target)
                generated = automatic_request(model, request, target)
                atomic_json(folder / 'generator_request.json', generated)
                if position:
                    value, prompt, refs, stats = condition(model, generated)
                    atomic_json(folder / 'condition_record.json', {**stats, 'references': refs})
                else:
                    value, prompt, refs = model.condition(generated, folder)
                pictures, pool = [], []
                for k, seed in enumerate(request['seeds']):
                    picture = model.pipe(prompt_embeds=value, height=512, width=512,
                        num_inference_steps=plan['steps'], guidance_scale=plan['guidance'], max_sequence_length=128,
                        generator=torch.Generator(device='cuda:0').manual_seed(seed)).images[0]
                    path = folder / f'candidate_{k:02d}.bmp'
                    picture.save(path, format='BMP')
                    pictures.append(picture)
                    pool.append({'candidate_index': k, 'seed': seed, 'base_example_id': request['example_id'],
                        'image': str(path), 'bytes': path.stat().st_size, 'prompt': prompt, 'actual_references': refs,
                        'generated_history': request['context']['history'], 'mapper_actually_used': bool(position),
                        'status': 'complete'})
                classifier = model.classify(pictures, target)
                grades = model.grade(request, [c['image'] for c in pool])
                if len(classifier) != len(pool) or len(grades) != len(pool):
                    raise ValueError('Every candidate requires its own classifier and visual grade')
                for candidate, cls, grade in zip(pool, classifier, grades):
                    candidate.update(classifier=cls, visual=grade)
                choice, ranking = select_local(request, pool)
                atomic_json(folder / 'candidates.json', pool)
                atomic_json(folder / 'selection.json', {'choice': choice, 'ranking': ranking})
                output = {'story_id': story['story_id'], 'clip_id': request['clip_id'],
                    'image': pool[choice]['image'], 'status': 'complete',
                    'kind': 'generated_story_frame', 'example_id': request['example_id']}
                atomic_json(folder / 'selected.json', output)
                history.append(output)
                completed += 1
                state = {'completed_frames': completed, 'total_frames': len(selected_stories)*4,
                         'physical_gpu': gpus[shard], 'seconds': round(time.monotonic()-started, 1)}
                atomic_json(status_path, {'status': 'running', **state})
                print(json.dumps(state), flush=True)
    atomic_json(status_path, {'status': 'complete', **state})


def summarize(values, mode):
    correct = tp = fp = fn = 0
    for row in values:
        target = set(row[mode + '_target_indices'])
        prediction = row[mode + '_predicted_indices']
        predicted = set(prediction or [])
        correct += prediction is not None and target == predicted
        tp += len(target & predicted)
        fp += len(predicted - target)
        fn += len(target - predicted)
    return {'frames': len(values), 'correct': correct, 'accuracy_percent': 100*correct/len(values),
            'micro_f1_percent': 200*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 100.}


def finish_causal(root):
    plan = restore_plan(root)
    manifest = rows(root / 'inputs/evaluation.jsonl')
    with (Path(plan['labels']) / 'labels.pkl').open('rb') as stream:
        characters = pickle.load(stream)
    with (Path(plan['labels']) / 'labels_bg.pkl').open('rb') as stream:
        backgrounds = pickle.load(stream)['label']
    values, missing = [], []
    for item in manifest:
        folder = root / 'frames' / item['example_id']
        selected, probabilities = None, None
        if (folder / 'selected.json').exists():
            request, pool = read(folder / 'request.json'), read(folder / 'candidates.json')
            choice, ranking = select_local(request, pool)
            if read(folder / 'selection.json') != {'choice': choice, 'ranking': ranking}:
                raise ValueError('Saved selection differs from the declared ranking inputs')
            selected = read(folder / 'selected.json')
            if selected['image'] != pool[choice]['image'] or selected['example_id'] != item['example_id']:
                raise ValueError('Selected output differs from the request')
            from PIL import Image
            with Image.open(selected['image']) as image:
                image.load()
            probabilities = pool[choice]['classifier']
        else:
            missing.append(item['example_id'])
        row = {**item, 'image': selected['image'] if selected else None,
               'generation_status': 'complete' if selected else 'missing'}
        for mode, labels, field in [('char',characters,'character'),('bg',backgrounds,'background')]:
            row[mode + '_target_indices'] = [i for i,v in enumerate(labels[item['clip_id']]) if v == 1]
            row[mode + '_predicted_indices'] = [i for i,p in enumerate(probabilities[field + '_probabilities']) if p > .5] if probabilities else None
            row[mode + '_exact_match'] = probabilities is not None and row[mode+'_target_indices'] == row[mode+'_predicted_indices']
        values.append(row)
    write_rows(root / 'predictions.jsonl', values)
    result = {'status': 'complete' if not missing else 'incomplete', 'variant': plan['variant'],
        'planned_frames': len(manifest), 'generated_frames': len(manifest)-len(missing), 'missing': missing,
        'development_validation': True, 'changed_reference_protocol': plan['variant'] == 'history_best4',
        'scores': {mode: summarize(values, mode) for mode in ('char','bg')},
        'score_scope': 'Exact character/background label sets; not source identity or story quality acceptance'}
    atomic_json(root / 'result.json', result)
    print(json.dumps(result, ensure_ascii=False))


def finish_raw(root):
    from raw_rank import candidate_rows
    from raw_prepare import make_views
    plan = restore_plan(root)
    views = make_views(root, candidate_rows(root))
    commands = []
    for name in views:
        commands.append((plan['gpus'][0], [plan['score_python'], '-B', ROOT / 'core/stage_b_score.py',
            '--manifest', root / 'inputs/evaluation.jsonl', '--generation', root / 'views' / name / 'generated',
            '--data', plan['labels'], '--classifiers', plan['classifiers'],
            '--output', root / 'views' / name / 'scores', '--task-label', 'Flintstones raw ' + name]))
    for command in commands:
        workers([command])
    result = {name: read(root/'views'/name/'scores/scores.json') for name in views}
    atomic_json(root / 'result.json', result)
    print(json.dumps({name: value['scores'] for name,value in result.items()}))


def run(root):
    plan = restore_plan(root)
    raw = plan.get('research_variant', plan['variant']).startswith('raw_')
    command = ROOT/'core/raw_generate.py' if raw else Path(__file__)
    commands = []
    for shard,gpu in enumerate(plan['gpus']):
        argv = [plan['python'], '-B', command]
        argv += ['--root', root, '--shard', str(shard), '--shards', str(len(plan['gpus']))] if raw else ['worker', '--output', root, '--shard', str(shard)]
        commands.append((gpu, argv))
    workers(commands)
    if raw and plan['candidates_per_frame'] > 1:
        workers([(plan['gpus'][0], [plan['score_python'], '-B', __file__, 'rank-classifier', '--output', root])])
        workers([(gpu, [plan['python'], '-B', __file__, 'rank-visual', '--output', root, '--shard', str(shard)])
                 for shard,gpu in enumerate(plan['gpus'])])
    (finish_raw if raw else finish_causal)(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','run','worker','finish','rank-classifier','rank-visual'])
    parser.add_argument('--config', type=Path, default=ROOT/'configs/metrics.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--variant', choices=VARIANTS, default='raw_best4')
    parser.add_argument('--stories', type=Path)
    parser.add_argument('--limit-stories', type=int)
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args)
        return
    if args.output is None:
        parser.error('--output is required')
    root = args.output.resolve()
    if args.action == 'run':
        run(root)
    elif args.action == 'worker':
        causal_worker(root, args.shard)
    elif args.action == 'finish':
        plan = restore_plan(root)
        (finish_raw if plan.get('research_variant', plan['variant']).startswith('raw_') else finish_causal)(root)
    elif args.action == 'rank-classifier':
        from raw_rank import classify
        classify(root)
    else:
        from raw_rank import visual
        plan = restore_plan(root)
        visual(root, args.shard, len(plan['gpus']), 2)


if __name__ == '__main__':
    main()
