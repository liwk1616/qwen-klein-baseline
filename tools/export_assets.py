"""List or copy the external assets needed by both baselines, preserving layout."""
import argparse
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def inventory(source, training=False):
    config = json.loads((ROOT/'configs/metrics.json').read_text())
    required = [config[key] for key in ('raw_stories', 'raw_seed_subset', 'raw_annotations',
                                       'stories', 'background_names')]
    required += [config['data']+'/reference_bank.json', config['labels']+'/labels.pkl',
                 config['labels']+'/labels_bg.pkl', config['bank_source']+'/official_character_features.safetensors']
    for key in ('character_checkpoint', 'raw_character_checkpoint'):
        required += [config[key]+'/'+name for name in
                     ('config.json', 'complete.json', 'adapter_model.safetensors', 'character_fusion.safetensors')]
    required += [config['mapper_checkpoint']+'/'+name for name in ('config.json','complete.json','mapper.safetensors')]
    required += [config['classifiers']+'/classifier_'+mode+'.safetensors' for mode in ('char','bg')]
    required += [config['model']+'/model_index.json']
    folders = [config['qwen']] + [config['model']+'/'+name for name in
                                ('transformer','text_encoder','tokenizer','vae','scheduler')]
    checkpoint = json.loads((source/config['raw_character_checkpoint']/'config.json').read_text())
    suffix = '/'+config['model']
    if not checkpoint['model'].endswith(suffix):
        raise ValueError('Unexpected recorded model location')
    recorded_root = Path(checkpoint['model'][:-len(suffix)])
    for ref in json.loads((source/config['data']/'reference_bank.json').read_text()):
        required.append(str(Path(ref['path']).relative_to(recorded_root)))
    if training:
        required += [config['data']+'/protocol.json', config['data']+'/teacher_conditioning.jsonl',
                     'data/downloads/flintstones.zip']
        folders += [config['data']+'/clips', config['data']+'/stories',
                    config['labels']+'/video_frames_sampled_4x', config['character_checkpoint']]
        # Reference adaptation requires the full C0 optimizer and RNG state.
        required.append(config['character_checkpoint']+'/optimizer.safetensors')
    for folder in folders:
        base = source/folder
        if not base.is_dir():
            raise FileNotFoundError(base)
        for directory, subdirs, names in os.walk(base, followlinks=True):
            subdirs[:] = [name for name in subdirs if name not in ('.cache','__pycache__','download_parts','.git')]
            for name in names:
                if name.startswith('download_') or name.endswith(('.lock','.partial')):
                    continue
                required.append(str((Path(directory)/name).relative_to(source)))
    records = []
    for relative in sorted(set(required)):
        path = source/relative
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        records.append({'path':relative, 'bytes':path.stat().st_size})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Existing QwenAnimeBaseline asset root')
    parser.add_argument('--pilot-cases', type=Path, help='Original pilot20 inputs/cases.json')
    parser.add_argument('--training', action='store_true', help='Include raw frames/masks and C0 resume state')
    parser.add_argument('--copy-to', type=Path, help='Explicitly copy to a NEW directory; omitted means list only')
    args = parser.parse_args()
    source = args.source.resolve()
    records = inventory(source, args.training)
    if args.pilot_cases and not args.pilot_cases.is_file():
        raise FileNotFoundError(args.pilot_cases)
    result = {'files':len(records), 'bytes':sum(r['bytes'] for r in records),
              'training_assets':args.training, 'records':records}
    if args.copy_to:
        destination = args.copy_to.resolve()
        if destination.is_relative_to(source):
            raise ValueError('Choose a destination outside the source asset tree')
        destination.mkdir(parents=True, exist_ok=False)
        for record in records:
            target = destination/'QwenAnimeBaseline'/record['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            # Dereference model component symlinks so the copy is self-contained.
            shutil.copyfile(source/record['path'], target)
        if args.pilot_cases:
            (destination/'pilot20').mkdir()
            shutil.copyfile(args.pilot_cases, destination/'pilot20/cases.json')
        (destination/'ASSET_INVENTORY.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({key:value for key,value in result.items() if key != 'records'}, indent=2))
    else:
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
