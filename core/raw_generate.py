"""Generate all declared independent candidates; never inspect evaluation labels."""
import argparse
import json
from asset_paths import loads as asset_loads
import os
from pathlib import Path
import time

from archive_io import atomic_json
from independent_candidate_protocol import PROTOCOL, bind_request
from runtime_io import rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--shards', type=int, default=4)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', DIFFUSERS_OFFLINE='1', TORCH_COMPILE_DISABLE='1')
    import torch
    from diffusers import Flux2KleinPipeline
    from peft import LoraConfig, set_peft_model_state_dict
    from safetensors import safe_open
    from safetensors.torch import load_file
    from char_klein import CharacterFusion
    from lora_config import LORA_MODULES
    torch.set_num_threads(4)
    root = args.root.resolve()
    plan = asset_loads((root/'run_plan.json').read_text())
    authorization = asset_loads((root/'authorization.json').read_text())
    if (plan['protocol'] != PROTOCOL or not authorization.get('independent_original_captions_allowed')
            or not authorization.get('fixed_reference_bank_every_frame_allowed')):
        raise ValueError('Independent candidate generation is not authorized for this run')
    requests = [r for r in rows(root/'inputs/requests.jsonl') if r['story_index'] % args.shards == args.shard]
    if not requests or not 0 <= args.shard < args.shards:
        raise ValueError('Invalid or empty generation shard')
    checkpoint = Path(plan['character_checkpoint'])
    config = asset_loads((checkpoint/'config.json').read_text())
    if not (checkpoint/'complete.json').is_file() or config['model'] != plan['model']:
        raise ValueError('Incomplete or unexpected inference checkpoint')
    directory = root/'candidates'/f'rank_{args.shard:03d}'
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'images').mkdir()
    pipe = Flux2KleinPipeline.from_pretrained(config['model'], torch_dtype=torch.bfloat16, local_files_only=True).to('cuda:0')
    if pipe.config.is_distilled:
        raise ValueError('Expected undistilled Klein Base')
    pipe.transformer.add_adapter(LoraConfig(r=config['rank'], lora_alpha=config['rank'],
        target_modules=LORA_MODULES, bias='none', lora_dropout=0.))
    loaded = set_peft_model_state_dict(pipe.transformer, load_file(checkpoint/'adapter_model.safetensors'), adapter_name='default')
    if loaded.unexpected_keys or any('lora_' in key for key in loaded.missing_keys):
        raise ValueError('Adapter loading failed')
    fusion = CharacterFusion().to('cuda:0').eval().requires_grad_(False)
    fusion.load_state_dict(load_file(checkpoint/'character_fusion.safetensors'), strict=True)
    for module in [pipe.transformer, pipe.text_encoder, pipe.vae]:
        module.eval().requires_grad_(False)
    bank_path = Path(plan['bank_source'])/'official_character_features.safetensors'
    with safe_open(bank_path, framework='pt', device='cuda:0') as source:
        meta = source.metadata()
        bank_names, bank_images = asset_loads(meta['names']), asset_loads(meta['images'])
        features = source.get_tensor('visual_features')
    bank = {name: (index, picture) for index, (name, picture) in enumerate(zip(bank_names, bank_images))}
    atomic_json(directory/'runtime.json', {'checkpoint': str(checkpoint), 'model': config['model'],
        'reference_feature_file': str(bank_path), 'steps': plan['steps'], 'guidance': plan['guidance'],
        'shard': args.shard, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'history_used': False, 'mapper_used': False, 'original_target_images_used': False,
        'evaluation_labels_used': False, 'prompt_variant': plan['variant'],
        'candidates_per_frame': plan['candidates_per_frame']})
    pipe.set_progress_bar_config(disable=True)
    started, completed = time.monotonic(), 0
    total = sum(len(r['seeds']) for r in requests)
    with torch.inference_mode(), (directory/'generated.jsonl').open('x') as stream:
        for request in requests:
            if (request['context'] != {'history': []} or request['variant'] != plan['variant']
                    or request['split'] != plan['split'] or request['protocol'] != PROTOCOL):
                raise ValueError('Request escaped the independent current-caption protocol')
            length, bindings = bind_request(pipe.tokenizer, request)
            if bindings != request['bindings'] or length != request['native_sequence_length']:
                raise ValueError('Actual token bindings differ from preflight')
            references = {r['name']: r['image'] for r in request['references']}
            if set(references) != {b['name'] for b in bindings}:
                raise ValueError('Not every selected reference is bound')
            actual, visual, positions = [], [], []
            for binding in bindings:
                name, token = binding['name'], binding['token_index']
                row, picture = bank[name]
                if picture != references[name]:
                    raise ValueError('Reference picture differs from the encoded bank')
                visual.append(features[row]); positions.append(token)
                actual.append({'name': name, 'image': picture, 'feature_file': str(bank_path),
                               'feature_row': row, 'text_token_index': token,
                               'caption_span': [binding['start'], binding['end']]})
            text, _ = pipe.encode_prompt(request['prompt'], device=torch.device('cuda:0'), max_sequence_length=length)
            visual = torch.stack(visual).unsqueeze(0) if visual else torch.zeros(1, 0, 2560, device='cuda:0', dtype=torch.bfloat16)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                condition = fusion(text, visual, torch.tensor([positions], device='cuda:0', dtype=torch.int64))
            if not bool(torch.isfinite(condition).all()):
                raise ValueError('Nonfinite generation condition')
            for k, seed in enumerate(request['seeds']):
                identifier = request['example_id']+f'__candidate_{k:02d}'
                picture = pipe(prompt_embeds=condition, height=512, width=512,
                    num_inference_steps=plan['steps'], guidance_scale=plan['guidance'], max_sequence_length=length,
                    generator=torch.Generator(device='cuda:0').manual_seed(seed)).images[0]
                path = directory/'images'/(identifier+'.bmp')
                picture.save(path, format='BMP')
                value = {key: request[key] for key in ['task_id', 'story_id', 'clip_id', 'target_position', 'target_caption', 'prompt']}
                value.update(index=request['index']*plan['candidates_per_frame']+k, example_id=identifier,
                    base_example_id=request['example_id'], frame_index=request['index'], candidate_index=k,
                    seed=seed, status='complete', kind='independent_story_frame_candidate', image=str(path),
                    bytes=path.stat().st_size, actual_references=actual, checkpoint=str(checkpoint),
                    reference_refresh=True, mapper_actually_used=False, generated_history=[],
                    ground_truth_images_seen=False, future_captions_seen=False, evaluation_labels_seen=False,
                    source_gate='fixed_bank_and_original_current_caption', variant=plan['variant'],
                    native_sequence_length=length)
                stream.write(json.dumps(value, ensure_ascii=False)+'\n'); stream.flush()
                completed += 1
                state = {'status': 'running', 'completed': completed, 'total': total,
                         'elapsed_seconds': round(time.monotonic()-started, 1)}
                atomic_json(directory/'status.json', state)
                if completed == 1 or completed % 8 == 0:
                    print(json.dumps(state), flush=True)
    atomic_json(directory/'status.json', {**state, 'status': 'complete'})


if __name__ == '__main__':
    main()
