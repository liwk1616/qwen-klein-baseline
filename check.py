"""CPU checks for a clean checkout, external assets and local model APIs."""
import argparse
import ast
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import sys

from runtime import ROOT, read, metrics_config, asset_settings


def source_check():
    files = [p for p in ROOT.rglob('*') if p.is_file() and
             not any(part in {'.git', '__pycache__', '.venv-qwen', '.venv-score', 'assets', 'outputs'}
                     for part in p.relative_to(ROOT).parts)]
    sources = [p for p in files if p.suffix == '.py']
    for path in sources:
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    for path in files:
        if path.suffix == '.json':
            json.loads(path.read_text(encoding='utf-8-sig'))
    return {'status': 'passed', 'python_sources': len(sources), 'model_inference': False}


def require_file(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError('Missing or empty asset: '+str(path))
    return path


def model_files(path, flux=False):
    path = Path(path)
    components = ('transformer', 'text_encoder', 'vae', 'tokenizer', 'scheduler') if flux else ('',)
    require_file(path/('model_index.json' if flux else 'config.json'))
    count = 0
    for name in components:
        folder = path/name
        if not folder.is_dir():
            raise FileNotFoundError('Missing model component or broken symlink: '+str(folder))
        shards = list(folder.glob('*.safetensors'))
        if name not in ('tokenizer', 'scheduler') and not shards:
            raise ValueError('No model weights in '+str(folder))
        for index in folder.glob('*.safetensors.index.json'):
            for shard in set(read(index)['weight_map'].values()):
                require_file(folder/shard)
        for shard in shards:
            require_file(shard)
            count += 1
    return count


def check_assets(runtime=False, training=False):
    config = metrics_config()
    settings = asset_settings()
    result = source_check()
    for key in ('python_qwen', 'python_flux'):
        require_file(settings[key])
    result['weight_files'] = {'flux_base': model_files(config['model'], True),
                              'qwen': model_files(config['qwen'])}
    if read(Path(config['model'])/'model_index.json').get('is_distilled', False):
        raise ValueError('Expected Klein Base, not the distilled model')
    for key in ('character_checkpoint', 'raw_character_checkpoint'):
        for name in ('config.json', 'complete.json', 'adapter_model.safetensors', 'character_fusion.safetensors'):
            require_file(Path(config[key])/name)
        checkpoint = read(Path(config[key])/'config.json')
        if Path(checkpoint['model']).resolve() != Path(config['model']).resolve():
            raise ValueError('Checkpoint/base model mismatch: '+key)
    for name in ('config.json', 'complete.json', 'mapper.safetensors'):
        require_file(Path(config['mapper_checkpoint'])/name)
    mapper = read(Path(config['mapper_checkpoint'])/'config.json')
    if Path(mapper['character_checkpoint']).resolve() != Path(config['character_checkpoint']).resolve():
        raise ValueError('The 8k mapper must be paired with the C0 25k character checkpoint')
    for mode in ('char', 'bg'):
        require_file(Path(config['classifiers'])/f'classifier_{mode}.safetensors')
    for key in ('raw_stories', 'raw_seed_subset', 'raw_annotations', 'background_names', 'stories'):
        require_file(config[key])
    from safetensors import safe_open
    from PIL import Image
    bank = read(Path(config['data'])/'reference_bank.json')
    for reference in bank:
        with Image.open(require_file(reference['path'])) as picture:
            picture.load()
    with safe_open(require_file(Path(config['bank_source'])/'official_character_features.safetensors'),
                   framework='pt', device='cpu') as features:
        from asset_paths import loads
        meta = features.metadata()
        names, images = loads(meta['names']), loads(meta['images'])
        shape = features.get_slice('visual_features').get_shape()
    if dict(zip(names, images)) != {r['name']:r['path'] for r in bank} or shape != [len(bank), 2560]:
        raise ValueError('Reference images and encoded feature rows do not match')
    from run_metrics import configured_stories
    cohorts = {key: configured_stories(config, key)[1] for key in ('raw_best4', 'history_best4')}
    with require_file(Path(config['labels'])/'labels.pkl').open('rb') as stream:
        characters = pickle.load(stream)
    with require_file(Path(config['labels'])/'labels_bg.pkl').open('rb') as stream:
        backgrounds = pickle.load(stream)
    if backgrounds['names'] != read(config['background_names']):
        raise ValueError('Background taxonomy order differs')
    for stories in cohorts.values():
        for story in stories:
            for target in story['targets']:
                cid = target['clip_id']
                if len(characters[cid]) != 7 or len(backgrounds['label'][cid]) != 323:
                    raise ValueError('Invalid evaluator label dimensions')
                for ref in target['initial_reference_bank']:
                    require_file(ref['image'])
    result.update(stories={key:len(value) for key,value in cohorts.items()},
                  reference_images=len(bank), paired_mapper=True, checkpoints_present=True)
    pilot = Path(settings['pilot_cases'])
    if pilot.is_file():
        from run_pilot import validate_cases
        cases = read(pilot)
        validate_cases(cases)
        result['pilot_cases'] = len(cases)
    else:
        result['pilot_cases'] = 'not installed; only the optional pilot command needs this file'
    if runtime:
        for path in (ROOT/'core').glob('*.py'):
            if path.stem == '__init__':
                continue
            module = importlib.import_module(path.stem)
            if not Path(module.__file__).resolve().is_relative_to(ROOT):
                raise ValueError('Project module imported from outside this checkout: '+path.stem)
        from transformers import AutoTokenizer, AutoProcessor, Qwen3_5ForConditionalGeneration
        from diffusers import Flux2KleinPipeline
        from peft import LoraConfig
        tokenizer = AutoTokenizer.from_pretrained(Path(config['model'])/'tokenizer', local_files_only=True)
        AutoProcessor.from_pretrained(config['qwen'], local_files_only=True, trust_remote_code=False)
        from independent_candidate_protocol import make_request, bind_request
        annotations = read(config['raw_annotations'])
        bank_dict = {r['name']:r['path'] for r in bank}
        lengths = set()
        for index, story in enumerate(cohorts['raw_best4']):
            for position, target in enumerate(story['targets']):
                req = make_request(index, story, position, annotations[target['clip_id']]['caption'], bank_dict, 'raw', 4)
                lengths.add(bind_request(tokenizer, req)[0])
        result['raw_native_sequence_lengths'] = sorted(lengths)
        result['runtime_imports'] = 'passed'
        result['versions'] = {name:version(name) for name in ('torch', 'torchvision', 'diffusers', 'transformers',
                                                            'peft', 'safetensors', 'numpy', 'Pillow')}
    if training:
        from storygptv_data_bundle import StoryGPTVDataBundle
        bundle = StoryGPTVDataBundle(config['data'])
        require_file(bundle.archive_path)
        import random
        try:
            sample = bundle.stage1(0, random.Random(20260919))
            result['training_sample'] = {'image_shape': list(sample['image'].shape),
                                         'mask_shape': list(sample['segmentation'].shape),
                                         'train_clips': len(bundle.by_split['train'])}
        finally:
            bundle.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-only', action='store_true')
    parser.add_argument('--runtime', action='store_true')
    parser.add_argument('--training', action='store_true')
    args = parser.parse_args()
    result = source_check() if args.source_only else check_assets(args.runtime, args.training)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
