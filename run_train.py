"""Launch the current aligned character/story trainer from explicit settings."""
import argparse
import json
import os
from pathlib import Path
import sys

from runtime import ROOT, read, gpu_ids, metrics_config, environment, configure_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog='Additional arguments are passed to core/train_aligned_klein.py; use that file --help for all training options.')
    parser.add_argument('--stage', choices=['character','reference','story'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/training.json')
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--prepare-only', action='store_true', help='Print the command without starting training')
    args, extra = parser.parse_known_args()
    configure_assets()
    assets = metrics_config()
    config = read(args.config)[args.stage]
    for key in ('data', 'model', 'qwen'):
        config[key] = assets[key]
    if args.stage == 'story':
        config['character_checkpoint'] = assets['character_checkpoint']
    if args.stage == 'reference' and args.resume is None:
        parser.error('Reference adaptation requires --resume with the full C0 25k training checkpoint')
    if args.resume:
        config['resume'] = str(args.resume.resolve())
    if args.output.exists():
        raise FileExistsError('Use a new training output directory')
    gpus = gpu_ids(args.gpus)
    command = [sys.executable, '-B', '-m', 'torch.distributed.run', '--standalone',
               '--nproc_per_node', str(len(gpus)), str(ROOT/'core/train_aligned_klein.py')]
    for name,value in config.items():
        flag = '--' + name.replace('_','-')
        if isinstance(value, bool):
            if value:
                command.append(flag)
        elif value is not None:
            command.extend([flag, str(value)])
    command.extend(['--output', str(args.output.resolve())])
    command.extend(extra[1:] if extra[:1] == ['--'] else extra)
    env = dict(environment(), CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)))
    if args.prepare_only:
        print(json.dumps({'CUDA_VISIBLE_DEVICES':env['CUDA_VISIBLE_DEVICES'],'command':command},indent=2))
        return
    os.execve(sys.executable, command, env)


if __name__ == '__main__':
    main()
