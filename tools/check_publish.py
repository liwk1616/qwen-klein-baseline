"""Inspect the proposed source tree before uploading it; no Git mutation."""
import ast
from pathlib import Path
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    problems, files, total = [], [], 0
    ignored = {'.git', '__pycache__', '.pytest_cache', '.venv-qwen', '.venv-score',
               'assets', 'data', 'models', 'checkpoints', 'outputs', 'runs', 'logs', '.vscode', '.idea'}
    for path in sorted(ROOT.rglob('*')):
        relative = path.relative_to(ROOT)
        if any(part in ignored for part in relative.parts) or path.name.endswith('.local.json'):
            continue
        if path.is_symlink():
            problems.append(str(relative)+': unexpected symbolic link')
            continue
        if not path.is_file():
            continue
        size = path.stat().st_size
        files.append(str(relative)); total += size
        if size > 5_000_000 or path.suffix.lower() in {'.safetensors','.pt','.pth','.ckpt','.pkl','.npy','.npz','.zip','.bin'}:
            problems.append(str(relative)+': model/data/binary or oversized artifact')
            continue
        if path.name.startswith('.env'):
            problems.append(str(relative)+': environment credentials file')
        text = path.read_text(encoding='utf-8-sig')
        # Construct the patterns to keep this audit script itself portable.
        if re.search('/'+'(?:media|home|root|mnt)/[^\\s]+', text):
            problems.append(str(relative)+': machine-specific absolute path')
        if re.search('(?:gh'+'p_|github_'+'pat_|sk-'+'[A-Za-z0-9]{20})', text):
            problems.append(str(relative)+': possible credential')
        if path.suffix == '.py':
            ast.parse(text, filename=str(relative))
    print(json.dumps({'status':'failed' if problems else 'passed', 'files':len(files),
                      'bytes':total, 'problems':problems}, ensure_ascii=False, indent=2))
    return int(bool(problems))


if __name__ == '__main__':
    sys.exit(main())
