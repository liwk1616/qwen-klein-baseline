import json
from asset_paths import loads as asset_loads
from pathlib import Path

def rows(path):
    return [asset_loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]

def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False)+'\n')
