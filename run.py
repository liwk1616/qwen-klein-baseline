#!/usr/bin/env python3
"""Portable launcher for the contract baseline and frozen history baseline."""
import argparse
import os
from pathlib import Path
import sys

from runtime import ROOT, asset_settings, configure_assets, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, help='Machine-local asset/interpreter configuration')
    parser.add_argument('component', choices=['metrics', 'pilot', 'train', 'check', 'score'])
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.assets:
        os.environ['BASELINE_ASSETS_CONFIG'] = str(args.assets.resolve())
    # Source-only validation works on a fresh clone with the standard library.
    if args.component == 'check' and '--source-only' in args.arguments:
        from check import source_check
        import json
        print(json.dumps(source_check(), ensure_ascii=False, indent=2))
        return
    if '--help' in args.arguments or '-h' in args.arguments:
        script = (ROOT/'core/stage_b_score.py' if args.component == 'score' else
                  ROOT/('check.py' if args.component == 'check' else 'run_'+args.component+'.py'))
        os.execve(sys.executable, [sys.executable, '-B', str(script), *args.arguments], environment())
    settings = asset_settings()
    configure_assets()
    if args.component == 'score':
        script, interpreter = ROOT/'core/stage_b_score.py', settings['python_flux']
    else:
        script = ROOT/('check.py' if args.component == 'check' else 'run_'+args.component+'.py')
        interpreter = settings['python_qwen']
    if not Path(interpreter).is_file():
        parser.error('Interpreter missing: '+str(interpreter)+'; edit configs/assets.local.json')
    os.execve(str(interpreter), [str(interpreter), '-B', str(script), *args.arguments], environment())


if __name__ == '__main__':
    main()
