"""Frozen pilot request construction and per-frame generation."""
from pathlib import Path
import json
import time
from archive_io import atomic_json as write
from asset_paths import loads
PROTOCOL = 'pilot20_exact_caption_schedules_qwen_klein_20261009_v1'
def read(path):
    return loads(Path(path).read_text())

def make_request(case, arm, position, selected):
    from independent_candidate_protocol import SEED_STRIDE
    from storygptv_protocol import causal_history
    targets = case['arms'][arm]
    target = targets[position]
    context = causal_history(case['story_id'], targets, selected, position)
    label = 'shared' if position == 0 else arm
    return {'index': case['case_index']*15+position, 'example_id': f"case{case['case_id']}_{label}_frame{position+1}",
        'task_id': f"pilot20_{case['case_id']}_{label}", 'story_id': case['story_id'],
        'clip_id': target['clip_id'], 'target_position': position+1, 'target_caption': target['caption'],
        'context': context, 'initial_reference_bank': target['initial_reference_bank'] if position == 0 else [],
        'seeds': [target['seed']+SEED_STRIDE*k for k in range(4)], 'protocol': PROTOCOL, 'split': target['split']}

def generate_frame(models, root, request):
    from current_bank import automatic_request
    from current_bank import condition
    from causal_candidate_protocol import select_local
    directory = root/'frames'/request['example_id']
    if (directory/'selected.json').is_file():
        if read(directory/'request.json') != request:
            raise ValueError('Resume request or selected history differs')
        output = read(directory/'selected.json')
        if not Path(output['image']).is_file():
            raise FileNotFoundError(output['image'])
        return output, False
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    write(directory/'request.json', request)
    target = models.resolve_for_ranking(request)
    generated = automatic_request(models, request, target)
    write(directory/'ranking_target.json', target)
    write(directory/'generator_request.json', generated)
    if request['target_position'] > 1:
        value, prompt, refs, stats = condition(models, generated)
        write(directory/'condition_record.json', {**stats, 'references': refs})
    else:
        value, prompt, refs = models.condition(generated, directory)
    torch = models.torch
    pictures, pool = [], []
    for k, seed in enumerate(request['seeds']):
        picture = models.pipe(prompt_embeds=value, height=models.plan['height'], width=models.plan['width'],
            num_inference_steps=models.plan['steps'], guidance_scale=models.plan['guidance'],
            max_sequence_length=128, generator=torch.Generator(device='cuda:0').manual_seed(seed)).images[0]
        path = directory/f'candidate_{k:02d}.bmp'
        picture.save(path, format='BMP')
        pictures.append(picture)
        pool.append({'candidate_index': k, 'seed': seed, 'base_example_id': request['example_id'],
            'image': str(path), 'bytes': path.stat().st_size, 'prompt': prompt, 'actual_references': refs,
            'generated_history': request['context']['history'],
            'mapper_actually_used': request['target_position'] > 1, 'status': 'complete'})
    classifiers = models.classify(pictures, target)
    grades = models.grade(request, [c['image'] for c in pool])
    for candidate, cls, visual in zip(pool, classifiers, grades):
        candidate.update(classifier=cls, visual=visual)
    choice, ranking = select_local(request, pool)
    write(directory/'candidates.json', pool)
    write(directory/'selection.json', {'choice': choice, 'ranking': ranking})
    output = {'story_id': request['story_id'], 'clip_id': request['clip_id'], 'image': pool[choice]['image'],
        'status': 'complete', 'kind': 'generated_story_frame', 'example_id': request['example_id'],
        'elapsed_seconds': round(time.monotonic()-started, 2)}
    pictures[choice].save(directory/'selected.jpg', quality=94)
    write(directory/'selected.json', output)
    print(json.dumps({'completed': request['example_id'], 'choice': choice,
        'seconds': output['elapsed_seconds'], 'grader_errors': sum(c['visual']['error'] is not None for c in pool)}), flush=True)
    return output, True
