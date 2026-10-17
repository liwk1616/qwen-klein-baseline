"""Small shared launcher helpers; model and data files stay external."""
import json
import os
from pathlib import Path
import subprocess
import time
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'core'))
from asset_paths import configure, loads, mappings


def read(path):
    return loads(Path(path).read_text(encoding='utf-8-sig'))


def asset_settings():
    default = ROOT/'configs/assets.local.json'
    if not default.exists():
        default = ROOT/'configs/assets.example.json'
    path = Path(os.environ.get('BASELINE_ASSETS_CONFIG', default)).resolve()
    values = json.loads(path.read_text(encoding='utf-8-sig'))
    for key in ('project_root', 'pilot_cases', 'python_qwen', 'python_flux'):
        expanded = os.path.expandvars(os.path.expanduser(values[key]))
        if '$' in expanded:
            raise ValueError('Unresolved environment variable in '+key)
        candidate = Path(expanded)
        values[key] = str((candidate if candidate.is_absolute() else path.parent/candidate).absolute())
    return values


def configure_assets():
    settings = asset_settings()
    root = Path(settings['project_root'])
    config = json.loads((ROOT/'configs/metrics.json').read_text())
    # Checkpoint configuration identifies the original project location. Moving
    # an asset tree does not require editing any weight metadata or old results.
    checkpoint = root/config['raw_character_checkpoint']/'config.json'
    if not checkpoint.is_file():
        checkpoint = root/config['character_checkpoint']/'config.json'
    if not checkpoint.is_file():
        raise FileNotFoundError('Install the baseline assets first: '+str(checkpoint))
    original = json.loads(checkpoint.read_text())['model']
    suffix = '/'+config['model']
    if not original.endswith(suffix):
        raise ValueError('Cannot infer recorded asset root from checkpoint model path')
    recorded = original[:-len(suffix)]
    configure([(recorded, str(root))])
    return settings


def metrics_config(path=None):
    settings = configure_assets()
    config = read(path or ROOT/'configs/metrics.json')
    path_keys = ('model', 'qwen', 'data', 'bank_source', 'character_checkpoint',
                 'mapper_checkpoint', 'classifiers', 'background_names', 'stories',
                 'raw_stories', 'raw_seed_subset', 'raw_annotations',
                 'raw_character_checkpoint', 'labels')
    for key in path_keys:
        candidate = Path(config[key])
        if not candidate.is_absolute():
            config[key] = str(Path(settings['project_root'])/candidate)
    return config


def environment():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', DIFFUSERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', TORCH_COMPILE_DISABLE='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    env.pop('PYTHONPATH', None)
    return env


def restore_plan(root):
    # Configure mappings before reading any recorded checkpoint/reference paths.
    plan = json.loads((Path(root)/'run_plan.json').read_text())
    configure(plan.get('path_mappings', []))
    return read(Path(root)/'run_plan.json')


def gpu_ids(value):
    values = [int(v) for v in value.split(',')]
    if not values or len(set(values)) != len(values) or min(values) < 0:
        raise ValueError('GPU IDs must be distinct nonnegative integers')
    return values


def workers(commands):
    """Wait for every worker and propagate failures without orphaned processes."""
    running = []
    env = environment()
    try:
        for gpu, command in commands:
            running.append(subprocess.Popen([str(x) for x in command],
                cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu))))
        while True:
            codes = [p.poll() for p in running]
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError('A worker failed; see its error above')
            if all(code == 0 for code in codes):
                return
            time.sleep(.5)
    finally:
        for p in running:
            if p.poll() is None:
                p.terminate()
        for p in running:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
