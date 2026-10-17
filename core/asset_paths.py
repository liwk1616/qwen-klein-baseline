"""Resolve recorded asset locations in memory; never rewrite original assets."""
import json
import os
from pathlib import Path


def mappings():
    return json.loads(os.environ.get('BASELINE_PATH_MAPPINGS', '[]'))


def relocate(value):
    if isinstance(value, dict):
        return {key: relocate(item) for key, item in value.items()}
    if isinstance(value, list):
        return [relocate(item) for item in value]
    if isinstance(value, str):
        for old, new in sorted(mappings(), key=lambda pair: -len(pair[0])):
            if value == old or value.startswith(old.rstrip('/') + '/'):
                return new.rstrip('/') + value[len(old.rstrip('/')):]
    return value


def loads(value, **kwargs):
    return relocate(json.loads(value, **kwargs))


def configure(pairs):
    normalized = [(str(Path(old)), str(Path(new))) for old, new in pairs]
    os.environ['BASELINE_PATH_MAPPINGS'] = json.dumps(normalized)

