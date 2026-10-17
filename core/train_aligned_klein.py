"""Fresh two-stage Klein adaptation using online, visit-specific encodings."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timedelta
import json
from asset_paths import loads as asset_loads
import os
from pathlib import Path
import time

from archive_io import atomic_json

LORA_MODULES = r'^(?:transformer_blocks\.\d+\.attn\.(?:to_q|to_k|to_v|add_q_proj|add_k_proj|add_v_proj|to_add_out|to_out\.0)|single_transformer_blocks\.\d+\.attn\.(?:to_qkv_mlp_proj|to_out))$'


def validate_resume_config(old, new, allow_step_extension=False, allow_world_size_change=False,
                           allow_reference_policy_change=False):
    old_policy, new_policy = old.get('reference_policy', 'crop'), new.get('reference_policy', 'crop')
    if allow_reference_policy_change:
        if (old['stage'] != 'character' or new['stage'] != 'character'
                or old_policy != 'crop' or new_policy != 'fixed_bank_mixture' or allow_world_size_change):
            raise ValueError('Only an explicit character crop-to-bank-mixture continuation is supported')
    elif old_policy != new_policy:
        raise ValueError('Reference-policy changes require explicit authorization in the run configuration')
    if old_policy == 'fixed_bank_mixture':
        for key in ('bank_probability', 'reference_choice_seed'):
            if old[key] != new[key]:
                raise ValueError(f'Resume reference configuration changed: {key}')
    layout_keys = ['batch_size', 'accumulation', 'world_size']
    if allow_world_size_change:
        if old['world_size'] == new['world_size']:
            raise ValueError('Explicit world-size migration requires a changed GPU count')
        if old['batch_size'] * old['accumulation'] * old['world_size'] != new['batch_size'] * new['accumulation'] * new['world_size']:
            raise ValueError('World-size migration must preserve the effective batch size')
    for key in ['protocol', 'stage', 'data', 'model', 'qwen', 'rank', 'seed', 'batch_size', 'accumulation',
                'world_size', 'learning_rate', 'fusion_learning_rate', 'warmup', 'attention_weight',
                'alignment_weight', 'mask_loss_probability', 'character_checkpoint', 'deterministic_algorithms']:
        if allow_world_size_change and key in layout_keys:
            continue
        if old[key] != new[key]:
            raise ValueError(f'Resume configuration changed: {key}')
    if old['steps'] != new['steps']:
        if not allow_step_extension or new['steps'] <= old['steps']:
            raise ValueError('Changing the total budget requires an explicit positive step extension')
    elif allow_step_extension:
        raise ValueError('An explicit step extension must increase the previous total budget')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['character', 'story'], required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--qwen', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--character-checkpoint', type=Path)
    p.add_argument('--resume', type=Path)
    p.add_argument('--allow-step-extension', action='store_true',
                   help='Explicitly increase a saved training budget while restoring all continuation state')
    p.add_argument('--allow-world-size-change', action='store_true',
                   help='Preserve optimizer and per-step data, explicitly reseed stochastic noise for a new GPU layout')
    p.add_argument('--reference-policy', choices=['crop', 'fixed_bank_mixture'], default='crop')
    p.add_argument('--bank-probability', type=float, default=.5)
    p.add_argument('--reference-choice-seed', type=int, default=20260921)
    p.add_argument('--allow-reference-policy-change', action='store_true',
                   help='Explicit crop-to-bank-mixture adaptation; retain all original continuation state')
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--stop-at', type=int)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--rank', type=int, default=32)
    p.add_argument('--learning-rate', type=float, default=5e-5)
    p.add_argument('--fusion-learning-rate', type=float, default=1e-4)
    p.add_argument('--attention-weight', type=float, default=.1)
    p.add_argument('--alignment-weight', type=float, default=.1)
    p.add_argument('--mask-loss-probability', type=float, default=.3)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--seed', type=int, default=20260919)
    a = p.parse_args()
    stop_at = a.steps if a.stop_at is None else a.stop_at
    if not 0 <= stop_at <= a.steps or min(a.batch_size, a.accumulation, a.steps, a.warmup) < 1:
        raise ValueError('Invalid training budget')
    if a.allow_step_extension and a.resume is None:
        raise ValueError('A training-budget extension requires a resume checkpoint')
    if a.allow_world_size_change and a.resume is None:
        raise ValueError('A world-size migration requires a resume checkpoint')
    if a.reference_policy != 'crop' and a.stage != 'character':
        raise ValueError('Reference mixture applies only to character training')
    if a.allow_reference_policy_change and a.resume is None:
        raise ValueError('Reference adaptation requires a resume checkpoint')
    if not 0 <= a.bank_probability <= 1:
        raise ValueError('Invalid bank probability')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', DIFFUSERS_OFFLINE='1',
                      TORCH_COMPILE_DISABLE='1', TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from diffusers import Flux2Transformer2DModel
    from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file, save_file
    from aligned_data import AlignedData, LayoutSampler
    from aligned_encoders import OnlineEncoders
    from char_klein import CharKlein, CharacterFusion
    from story_klein_mapper import StoryConditionMapper
    rank = int(os.environ.get('RANK', 0))
    local = int(os.environ.get('LOCAL_RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    torch.set_num_threads(4)
    torch.cuda.set_device(local)
    torch.manual_seed(a.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda', local)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    root = a.output.resolve()
    if rank == 0:
        root.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    if asset_loads((a.model/'model_index.json').read_text()).get('is_distilled', False):
        raise ValueError('This experiment requires the official undistilled Base model')
    encoder = OnlineEncoders(a.model, a.qwen, a.data, device, a.stage)
    data = AlignedData(a.data, encoder.pipe.tokenizer)
    count = len(data.bundle.by_split['train'] if a.stage == 'character' else data.bundle.stories['train'])
    if count != (20656 if a.stage == 'character' else 19738):
        raise ValueError('Full released train population is required')
    transformer = Flux2Transformer2DModel.from_pretrained(a.model, subfolder='transformer',
                     torch_dtype=torch.bfloat16, local_files_only=True)
    transformer.requires_grad_(False)
    transformer.add_adapter(LoraConfig(r=a.rank, lora_alpha=a.rank, target_modules=LORA_MODULES, bias='none', lora_dropout=0.))
    transformer.enable_gradient_checkpointing()
    if a.stage == 'character':
        if a.character_checkpoint:
            raise ValueError('Fresh character stage forbids any character initialization checkpoint')
        for parameter in transformer.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        model = CharKlein(transformer).to(device)
        lora = [p for p in transformer.parameters() if p.requires_grad]
        groups = [{'params': lora, 'lr': a.learning_rate}, {'params': model.fusion.parameters(), 'lr': a.fusion_learning_rate}]
        fusion = model.fusion
    else:
        if a.character_checkpoint is None:
            raise ValueError('Story stage requires a newly selected character checkpoint')
        source = a.character_checkpoint.resolve()
        config = asset_loads((source/'config.json').read_text())
        if not (source/'complete.json').is_file() or config.get('protocol') != 'storygptv_dynamic_fresh_v1':
            raise ValueError('Story stage cannot load an old fixed-frame character run')
        if Path(config['data']).resolve() != a.data.resolve() or Path(config['model']).resolve() != a.model.resolve() or config['rank'] != a.rank:
            raise ValueError('Character model/data configuration differs')
        result = set_peft_model_state_dict(transformer, load_file(source/'adapter_model.safetensors'), adapter_name='default')
        if result.unexpected_keys or any('lora_' in k for k in result.missing_keys):
            raise ValueError('Character adapter load failed')
        transformer.requires_grad_(False).to(device).train()
        fusion = CharacterFusion().to(device).eval().requires_grad_(False)
        fusion.load_state_dict(load_file(source/'character_fusion.safetensors'), strict=True)
        model = StoryConditionMapper().to(device)
        groups = [{'params': model.parameters(), 'lr': a.learning_rate}]
    engine = DistributedDataParallel(model, device_ids=[local], broadcast_buffers=False) if world > 1 else model
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(groups, betas=(.9, .999), weight_decay=.01, foreach=False)
    total = sum(p.numel() for p in transformer.parameters()) + sum(p.numel() for p in fusion.parameters()) + sum(encoder.counts.values())
    # Count the intended complete inference system even before stage 2 exists.
    mapper_count = sum(p.numel() for p in model.parameters()) if a.stage == 'story' else 26092032
    total += mapper_count
    if total > 13000000000:
        raise ValueError('Combined inference parameter budget exceeds 13B')
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(protocol='storygptv_dynamic_fresh_v1', world_size=world,
                  effective_batch_size=a.batch_size*a.accumulation*world, train_examples=count,
                  trainable_parameters=sum(p.numel() for _, p in parameters), combined_parameters=total,
                  native_sequence_length=128, initialization='Official pretrained models; fresh task modules; only same-run resumes',
                  deterministic_algorithms=True, cublas_workspace_config=':4096:8',
                  online_encoders=True, old_feature_cache_used=False, caption_profile='official_evaluation_rule',
                  character_caption='Joined source tokens with native style suffix; unconditional means actual empty prompt',
                  condition_probabilities={'unconditional': .1, 'text_only': .1, 'character_augmented': .8},
                  teacher_policy='Official fixed bank and newly trained character fusion; teacher only',
                  history_policy='Original preceding images in training; generated preceding images in evaluation',
                  sampling='Rank-sharded epoch permutations; visit-derived frame/target/augmentation seed',
                  public_code_adaptations=['Qwen/Klein encoders and external mapper', 'Fix female pronouns and unconditional/text dropout',
                       'Omit invisible/invalid/ambiguous reference bindings', 'Keep existing image flow, attention and alignment objectives'],
                  test_used_for_training_or_selection=False, classifier_used_for_training=False)
    logical_layout = {'world_size': world, 'batch_size': a.batch_size, 'accumulation': a.accumulation}
    rng = torch.Generator(device=device).manual_seed(a.seed + rank*100)
    reference_mixer = None
    if a.reference_policy == 'fixed_bank_mixture':
        from character_reference_training import ReferenceMixer, ALIASES
        reference_mixer = ReferenceMixer(encoder.bank_rows, a.bank_probability, a.reference_choice_seed, rank)
        config.update(reference_aliases=ALIASES, reference_choice_unit='Distinct eligible identity per frame visit',
                      reference_preprocessing='Same fit_256_neutral_padding bank encoding as inference; crops unchanged')
    initial_step, visits = 0, 0
    coverage = {'clip_frames': {}, 'story_targets': {}, 'condition_counts': {}, 'target_exposures': 0}
    historical_coverage = []
    if a.resume:
        source = a.resume.resolve()
        if not (source/'complete.json').is_file():
            raise ValueError('Incomplete resume checkpoint')
        old = asset_loads((source/'config.json').read_text())
        validate_resume_config(old, config, a.allow_step_extension, a.allow_world_size_change,
                               a.allow_reference_policy_change)
        logical_layout = old.get('sampler_layout', {k: old[k] for k in logical_layout})
        if old['parameter_names'] != [name for name, _ in parameters]:
            raise ValueError('Optimizer parameter order differs')
        initial_step = old['step']
        if initial_step >= stop_at:
            raise ValueError('Resume must advance this new run')
        if a.stage == 'character':
            result = set_peft_model_state_dict(transformer, load_file(source/'adapter_model.safetensors'), adapter_name='default')
            if result.unexpected_keys or any('lora_' in k for k in result.missing_keys):
                raise ValueError('Adapter restore failed')
            model.fusion.load_state_dict(load_file(source/'character_fusion.safetensors'), strict=True)
        else:
            model.load_state_dict(load_file(source/'mapper.safetensors'), strict=True)
        if initial_step:
            states = load_file(source/'optimizer.safetensors')
            for i, (_, parameter) in enumerate(parameters):
                optimizer.state[parameter] = {key: states[f'parameter_{i}.{key}'].to('cpu' if key == 'step' else device)
                                              for key in ['step', 'exp_avg', 'exp_avg_sq']}
                if optimizer.state[parameter]['exp_avg'].shape != parameter.shape:
                    raise ValueError('Optimizer state shape differs')
        history_path = source/'historical_training_coverage.json'
        if history_path.is_file():
            historical_coverage = asset_loads(history_path.read_text())
        migration_seed = None
        if a.allow_world_size_change:
            old_coverage = [asset_loads((source/f'training_coverage_rank_{r:03d}.json').read_text())
                            for r in range(old['world_size'])]
            for item in old_coverage:
                if item.get('rank_visits', item['target_exposures']) != initial_step*old['batch_size']*old['accumulation']:
                    raise ValueError('Source rank sampling progress is incomplete')
            historical_coverage.append({'source_checkpoint': str(source), 'through_step': initial_step,
                                       'world_size': old['world_size'], 'rank_coverage': old_coverage})
            # Rank-local noise streams cannot be resumed verbatim with a new
            # batch layout. Record this change; optimizer and input samples do
            # retain their completed global-step position.
            migration_seed = a.seed + initial_step*1000003 + rank*1009
            rng.manual_seed(migration_seed)
            torch.manual_seed(migration_seed)
            visits = initial_step*a.accumulation*a.batch_size
            coverage.update(coverage_start_step=initial_step, rank_visits=visits,
                            scope='Actual rank exposures since the recorded GPU-layout migration')
        else:
            saved = load_file(source/f'rng_rank_{rank:03d}.safetensors')
            rng.set_state(saved['sampling'])
            torch.set_rng_state(saved['torch_cpu'])
            torch.cuda.set_rng_state(saved['torch_cuda'], device)
            if reference_mixer is not None and old.get('reference_policy') == 'fixed_bank_mixture':
                reference_mixer.rng.set_state(saved['reference_choice'])
            coverage = asset_loads((source/f'training_coverage_rank_{rank:03d}.json').read_text())
            visits = coverage.get('rank_visits', coverage['target_exposures'])
        if visits != initial_step*a.accumulation*a.batch_size:
            raise ValueError('Saved visit counter differs from completed optimizer steps')
        if coverage['target_exposures'] != (initial_step-coverage.get('coverage_start_step', 0))*a.accumulation*a.batch_size:
            raise ValueError('Rank coverage differs from its recorded accounting interval')
        atomic_json(root/f'resume_rank_{rank:03d}.json', {
            'status': 'restored', 'source_checkpoint': str(source), 'initial_step': initial_step,
            'previous_total_steps': old['steps'], 'total_steps': a.steps,
            'budget_extended': a.steps > old['steps'], 'rank': rank, 'rank_visits_restored': visits,
            'optimizer_parameter_states_restored': len(optimizer.state),
            'saved_rng_streams_restored': [] if a.allow_world_size_change else ['sampling', 'torch_cpu', 'torch_cuda'],
            'world_size_changed': a.allow_world_size_change, 'previous_world_size': old['world_size'],
            'world_size': world, 'new_noise_seed': migration_seed,
            'reference_policy_changed': a.allow_reference_policy_change,
            'reference_choice_state': ('restored' if old.get('reference_policy') == 'fixed_bank_mixture'
                                       else 'new independent stream') if reference_mixer is not None else None,
            'sampler_layout': logical_layout, 'same_optimizer_step_inputs_preserved': True,
            'learning_rate_schedule': 'Original global-step warmup then constant learning rates'})
    config['initial_step'] = initial_step
    config['sampler_layout'] = logical_layout
    config['sampling'] = 'Continuous complete epoch permutations; retain original logical-rank inputs after physical GPU migration'
    config['coverage_start_step'] = coverage.get('coverage_start_step', 0)
    if reference_mixer is not None:
        coverage.setdefault('reference_alignment', {'start_step': initial_step, 'counts': {}})
    sampler = LayoutSampler(count, rank, world, a.batch_size, a.accumulation, a.seed, logical_layout)
    if rank == 0:
        atomic_json(root/'run_config.json', config)
        atomic_json(root/'parameter_budget.json', {'combined_parameters': total, 'limit': 13000000000,
                    'includes_full_qwen_and_future_mapper': True, 'frozen_components': encoder.counts})
        save_file({'visual_features': encoder.bank_features.cpu().contiguous()}, root/'official_character_features.safetensors',
                  metadata={'names': json.dumps([r['name'] for r in encoder.bank_rows]),
                            'images': json.dumps([r['path'] for r in encoder.bank_rows]),
                            'visual_encoder': str(a.qwen.resolve()), 'preprocessing': 'fit_256_neutral_padding'})
    started = time.monotonic()
    def emit(event, **kwargs):
        if rank == 0:
            value = {'event': event, 'stage': a.stage, 'elapsed_seconds': round(time.monotonic()-started, 2),
                     'updated_at': datetime.now().astimezone().isoformat(), **kwargs}
            atomic_json(root/'status.json', value)
            with (root/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(value)+'\n')
            print(json.dumps(value), flush=True)

    def save(step):
        dest = root/f'checkpoint_{step:06d}'
        if rank == 0:
            dest.mkdir()
            if a.stage == 'character':
                save_file({k: v.detach().cpu().contiguous() for k, v in get_peft_model_state_dict(transformer).items()}, dest/'adapter_model.safetensors')
                save_file({k: v.detach().cpu().contiguous() for k, v in fusion.state_dict().items()}, dest/'character_fusion.safetensors')
            else:
                save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, dest/'mapper.safetensors')
            states = {f'parameter_{i}.{key}': value.detach().cpu().contiguous()
                      for i, (_, parameter) in enumerate(parameters) for key, value in optimizer.state[parameter].items()}
            if states:
                save_file(states, dest/'optimizer.safetensors')
            atomic_json(dest/'config.json', {**config, 'step': step, 'target_exposures': visits*world,
                                          'parameter_names': [name for name, _ in parameters]})
            if historical_coverage:
                atomic_json(dest/'historical_training_coverage.json', historical_coverage)
        if world > 1:
            dist.barrier()
        rng_states = {'sampling': rng.get_state().cpu().contiguous(), 'torch_cpu': torch.get_rng_state().contiguous(),
                      'torch_cuda': torch.cuda.get_rng_state(device).cpu().contiguous()}
        if reference_mixer is not None:
            rng_states['reference_choice'] = reference_mixer.rng.get_state().contiguous()
        save_file(rng_states, dest/f'rng_rank_{rank:03d}.safetensors')
        atomic_json(dest/f'training_coverage_rank_{rank:03d}.json', coverage)
        if world > 1:
            dist.barrier()
        if rank == 0:
            atomic_json(dest/'complete.json', {'step': step, 'world_size': world, 'files': sorted(p.name for p in dest.iterdir())})
        emit('checkpoint', step=step, path=str(dest))

    emit('initialized', initial_step=initial_step, target_steps=a.steps, stop_at=stop_at,
         effective_batch_size=config['effective_batch_size'], combined_parameters=total, fresh=a.resume is None)
    if a.resume is None:
        save(0)
    train_start = time.monotonic()
    stream = (root/f'sampling_rank_{rank:03d}.jsonl').open('x')
    try:
        for step in range(initial_step+1, stop_at+1):
            optimizer.zero_grad(set_to_none=True)
            for group, maximum in zip(optimizer.param_groups, [a.learning_rate, a.fusion_learning_rate]):
                group['lr'] = maximum*min(step/a.warmup, 1.)
            values = torch.zeros(3, dtype=torch.float32, device=device)
            encoded_seconds, model_seconds = 0., 0.
            for micro in range(a.accumulation):
                tick = time.monotonic()
                records = []
                for _ in range(a.batch_size):
                    index, seed, logical_rank, logical_visit = sampler.sample(visits)
                    sample = data.character(index, seed) if a.stage == 'character' else data.story(index, seed)
                    if reference_mixer is not None:
                        reference_mixer.apply(sample, data.bundle.annotations[sample['metadata']['clip_id']],
                                              coverage['reference_alignment']['counts'])
                    meta = sample['metadata']
                    stream.write(json.dumps({'step': step, 'microstep': micro, 'rank_visit': visits,
                                             'logical_rank': logical_rank, 'logical_visit': logical_visit,
                                             'dataset_index': index, **meta})+'\n')
                    target = meta if a.stage == 'character' else meta['target']
                    frame_counts = coverage['clip_frames'].setdefault(target['clip_id'], [0]*5)
                    frame_counts[target['frame_index']] += 1
                    if a.stage == 'character':
                        kind = meta['condition_kind']
                        coverage['condition_counts'][kind] = coverage['condition_counts'].get(kind, 0)+1
                    else:
                        counts = coverage['story_targets'].setdefault(meta['story_id'], [0]*3)
                        counts[meta['target_position']-2] += 1
                    records.append(sample)
                    visits += 1
                coverage['rank_visits'] = visits
                coverage['target_exposures'] = visits-coverage.get('coverage_start_step', 0)*a.batch_size*a.accumulation
                batch = encoder.character(records) if a.stage == 'character' else encoder.story(records)
                if not all(bool(torch.isfinite(v).all()) for v in batch.values()):
                    raise ValueError('Nonfinite online feature encoding')
                noise = torch.randn(batch['target'].shape, dtype=torch.bfloat16, device=device, generator=rng)
                sigma = torch.randn((a.batch_size,), device=device, generator=rng).sigmoid().clamp(.001, .999)
                encoded_seconds += time.monotonic()-tick
                tick = time.monotonic()
                sync = nullcontext() if world == 1 or micro == a.accumulation-1 else engine.no_sync()
                with sync, torch.autocast('cuda', dtype=torch.bfloat16):
                    if a.stage == 'character':
                        mask_only = (torch.rand((a.batch_size,), device=device, generator=rng) < a.mask_loss_probability) & (batch['indices'] >= 0).any(1)
                        loss, velocity, auxiliary = engine(**batch, noise=noise, sigma=sigma, mask_only=mask_only, attention_weight=a.attention_weight)
                    else:
                        with torch.no_grad():
                            teacher = fusion(batch['teacher_text'], batch['visual'], batch['indices'])
                        condition = engine(batch['qwen_states'], batch['current'])
                        alignment = ((condition.float()-teacher.float()).square().mean((1, 2)) / teacher.float().square().mean((1, 2)).clamp_min(1.)).mean()
                        noisy = ((1-sigma[:, None, None])*batch['target']+sigma[:, None, None]*noise).to(torch.bfloat16)
                        pred = transformer(hidden_states=noisy, encoder_hidden_states=condition, timestep=sigma,
                                           img_ids=batch['image_ids'], txt_ids=batch['text_ids'], return_dict=False)[0]
                        velocity = (pred.float()-(noise.float()-batch['target'].float())).square().mean()
                        loss, auxiliary = velocity+a.alignment_weight*alignment, alignment.detach()
                    if not bool(torch.isfinite(loss)):
                        raise ValueError('Nonfinite training objective')
                    (loss/a.accumulation).backward()
                values += torch.stack([loss.detach(), velocity.detach(), auxiliary.detach()])/a.accumulation
                model_seconds += time.monotonic()-tick
                del batch, records, loss
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in parameters], 1., error_if_nonfinite=True)
            optimizer.step()
            if world > 1:
                dist.all_reduce(values)
            values /= world
            stream.flush()
            if step <= initial_step+3 or step % 10 == 0:
                elapsed = time.monotonic()-train_start
                emit('training', step=step, target_steps=a.steps, combined_loss=float(values[0]), velocity_mse=float(values[1]),
                     auxiliary_loss=float(values[2]), gradient_norm=float(norm), target_exposures=visits*world,
                     unique_clips_this_rank=len(coverage['clip_frames']), unique_stories_this_rank=len(coverage['story_targets']),
                     seconds_per_step=round(elapsed/(step-initial_step), 3), encoding_seconds_last_step=round(encoded_seconds, 3),
                     forward_backward_seconds_last_step=round(model_seconds, 3),
                     learning_rates=[group['lr'] for group in optimizer.param_groups],
                     reference_counts_this_rank=coverage.get('reference_alignment', {}).get('counts'),
                     peak_memory_gib=round(torch.cuda.max_memory_allocated(device)/2**30, 3))
            if step % a.save_every == 0 or step == stop_at or step == 100:
                save(step)
        emit('segment_complete', status='complete', step=stop_at, final_checkpoint=str(root/f'checkpoint_{stop_at:06d}'))
    finally:
        stream.close()
        data.bundle.close()
        if world > 1:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
