"""Export, install, or publish the two frozen baselines' inference weights."""
import argparse
import getpass
import http.client
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import tempfile
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = 'liwk1616/qwen-klein-baseline'
TAG = 'weights-v1'
FORMAT = 'qwen-klein-frozen-weights-v1'
VARIANTS = ('raw_best4', 'history_best4')
DOCUMENTS = ('MODEL_CARD.md', 'LICENSE-WEIGHTS', 'NOTICE-WEIGHTS.md')
SLOTS = (
    ('raw_best4', 'raw_character_checkpoint', 'adapter_model.safetensors', 'raw-d1-lora.safetensors'),
    ('raw_best4', 'raw_character_checkpoint', 'character_fusion.safetensors', 'raw-d1-fusion.safetensors'),
    ('history_best4', 'character_checkpoint', 'adapter_model.safetensors', 'history-c0-lora.safetensors'),
    ('history_best4', 'character_checkpoint', 'character_fusion.safetensors', 'history-c0-fusion.safetensors'),
    ('history_best4', 'mapper_checkpoint', 'mapper.safetensors', 'history-mapper-8k.safetensors'),
)


def settings():
    return json.loads((ROOT/'configs/metrics.json').read_text())


def weight_specs():
    config = settings()
    return [dict(variant=variant, filename=filename, path=config[key]+'/'+name)
            for variant, key, name, filename in SLOTS]


def metadata_specs():
    config = settings()
    rows = []
    for variant, key in (('raw_best4', 'raw_character_checkpoint'),
                         ('history_best4', 'character_checkpoint'),
                         ('history_best4', 'mapper_checkpoint')):
        names = ['config.json', 'complete.json']
        if key == 'raw_character_checkpoint':
            names.append('component_layouts.json')
        rows.extend(dict(variant=variant, path=config[key]+'/'+name) for name in names)
    return rows


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2)+'\n'


def within(path, root):
    try:
        Path(path).relative_to(root)
        return True
    except ValueError:
        return False


def tensor_layout(path):
    """Check readable safetensors metadata and payload lengths, without hashes."""
    path = Path(path)
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('Truncated safetensors header: '+path.name)
        length = struct.unpack('<Q', prefix)[0]
        if not 2 <= length <= 4_000_000:
            raise ValueError('Invalid safetensors header length: '+path.name)
        header = json.loads(stream.read(length))
    widths = {'BOOL': 1, 'U8': 1, 'I8': 1, 'I16': 2, 'U16': 2,
              'F16': 2, 'BF16': 2, 'I32': 4, 'U32': 4, 'F32': 4,
              'I64': 8, 'U64': 8, 'F64': 8}
    rows = {name: info for name, info in header.items() if name != '__metadata__'}
    if not rows:
        raise ValueError('No tensors: '+path.name)
    end = 0
    for info in sorted(rows.values(), key=lambda item: item['data_offsets']):
        start, stop = info['data_offsets']
        shape = info['shape']
        if any(not isinstance(n, int) or n < 0 for n in shape):
            raise ValueError('Invalid tensor shape: '+path.name)
        if start != end or stop-start != math.prod(shape)*widths[info['dtype']]:
            raise ValueError('Invalid tensor offsets: '+path.name)
        end = stop
    if path.stat().st_size != 8+length+end:
        raise ValueError('Truncated or oversized tensor payload: '+path.name)
    return rows


def validate_manifest(manifest):
    if manifest.get('format') != FORMAT or manifest.get('license') != 'Apache-2.0':
        raise ValueError('Unsupported weight release format or license')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', manifest['repository']):
        raise ValueError('Invalid GitHub repository')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', manifest['tag']):
        raise ValueError('Invalid release tag')
    for key, specs in (('weights', weight_specs()), ('metadata', metadata_specs())):
        records = manifest[key]
        if len(records) != len(specs):
            raise ValueError('Incomplete or unexpected '+key)
        for record, spec in zip(records, specs):
            if any(record.get(k) != v for k, v in spec.items()):
                raise ValueError('Unexpected weight role, variant, or destination')
            if not isinstance(record['bytes'], int) or record['bytes'] <= 0:
                raise ValueError('Invalid file size')
            if key == 'metadata':
                if len(record['text'].encode('utf-8')) != record['bytes']:
                    raise ValueError('Metadata size mismatch')
                json.loads(record['text'])
    config = settings()
    original = {row['path']: json.loads(row['text']) for row in manifest['metadata']}
    raw = original[config['raw_character_checkpoint']+'/config.json']
    character = original[config['character_checkpoint']+'/config.json']
    mapper = original[config['mapper_checkpoint']+'/config.json']
    suffix = '/'+config['model']
    if not raw['model'].endswith(suffix):
        raise ValueError('Unexpected recorded base model path')
    recorded = raw['model'][:-len(suffix)]
    if (character['model'] != raw['model'] or mapper['model'] != raw['model']
            or mapper['character_checkpoint'] != recorded+'/'+config['character_checkpoint']
            or character['step'] != 25000 or mapper['step'] != 8000):
        raise ValueError('Frozen C0/mapper pairing or training step differs')
    return manifest


def validate_directory(directory):
    directory = Path(directory)
    manifest = validate_manifest(json.loads((directory/'WEIGHTS_MANIFEST.json').read_text()))
    for record in manifest['weights']:
        path = directory/record['filename']
        if path.stat().st_size != record['bytes'] or tensor_layout(path) != record['tensors']:
            raise ValueError('Weight size or tensor layout differs: '+path.name)
    for name in DOCUMENTS+('RELEASE_NOTES.md',):
        if not (directory/name).is_file() or (directory/name).stat().st_size == 0:
            raise FileNotFoundError(directory/name)
    return manifest


def export_weights(source, destination, repository=REPOSITORY, tag=TAG):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if within(destination, source) or within(destination, ROOT):
        raise ValueError('Export outside the original assets and the Git source tree')
    manifest = dict(format=FORMAT, repository=repository, tag=tag, license='Apache-2.0',
                    inference_only=True, training_resume_supported=False,
                    checks='File sizes, explicit tensor layouts, readable JSON; no content fingerprints.',
                    weights=[], metadata=[])
    for spec in weight_specs():
        path = source/spec['path']
        manifest['weights'].append(dict(spec, bytes=path.stat().st_size, tensors=tensor_layout(path)))
    for spec in metadata_specs():
        payload = (source/spec['path']).read_bytes()
        manifest['metadata'].append(dict(spec, bytes=len(payload), text=payload.decode('utf-8')))
    validate_manifest(manifest)
    destination.mkdir(parents=True, exist_ok=False)
    for record in manifest['weights']:
        shutil.copyfile(source/record['path'], destination/record['filename'])
    (destination/'WEIGHTS_MANIFEST.json').write_text(json_text(manifest), encoding='utf-8')
    for name in DOCUMENTS:
        origin = ROOT/'docs'/name if name == 'MODEL_CARD.md' else ROOT/name
        shutil.copyfile(origin, destination/name)
    shutil.copyfile(ROOT/'docs/WEIGHT_RELEASE_NOTES.md', destination/'RELEASE_NOTES.md')
    validate_directory(destination)
    return manifest


def stream_copy(source, target, expected):
    total = 0
    while True:
        block = source.read(min(1024*1024, expected-total+1))
        if not block:
            break
        total += len(block)
        if total > expected:
            raise ValueError('Downloaded file exceeds its recorded length')
        target.write(block)
    if total != expected:
        raise ValueError('Downloaded file is incomplete')


def install_weights(destination, variant='all', source_dir=None, repository=REPOSITORY, tag=TAG):
    if variant not in (*VARIANTS, 'all'):
        raise ValueError('Unknown baseline variant')
    base = 'https://github.com/'+repository+'/releases/download/'+quote(tag, safe='')+'/'

    def open_asset(name):
        if source_dir is not None:
            return (Path(source_dir)/name).open('rb')
        return urlopen(Request(base+quote(name), headers={'User-Agent': 'qwen-klein-baseline'}), timeout=60)

    with open_asset('WEIGHTS_MANIFEST.json') as stream:
        manifest_bytes = stream.read(2_000_001)
    if len(manifest_bytes) > 2_000_000:
        raise ValueError('Oversized manifest')
    manifest = validate_manifest(json.loads(manifest_bytes))
    if manifest['repository'] != repository or manifest['tag'] != tag:
        raise ValueError('Downloaded a different release')
    selected = [row for row in manifest['weights'] if variant == 'all' or row['variant'] == variant]
    metadata = [row for row in manifest['metadata'] if variant == 'all' or row['variant'] == variant]
    destination = Path(destination).resolve()
    doc_folder = 'weight_release_notes/'+tag+'/'+variant
    planned = [row['path'] for row in selected+metadata]
    planned += [doc_folder+'/'+name for name in (*DOCUMENTS, 'WEIGHTS_MANIFEST.json')]
    for relative in planned:
        path = destination/relative
        if not within(path.resolve(), destination):
            raise ValueError('Destination escapes the asset root')
        if path.exists() or path.is_symlink():
            raise FileExistsError('Refusing to overwrite '+str(path))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.baseline-weights-', dir=destination.parent) as temporary:
        stage = Path(temporary)
        for row in selected:
            path = stage/row['path']
            path.parent.mkdir(parents=True, exist_ok=True)
            with open_asset(row['filename']) as stream, path.open('xb') as target:
                stream_copy(stream, target, row['bytes'])
            if tensor_layout(path) != row['tensors']:
                raise ValueError('Downloaded tensor layout differs: '+row['filename'])
            print('Prepared '+row['filename'], flush=True)
        for row in metadata:
            path = stage/row['path']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(row['text'].encode('utf-8'))
        (stage/doc_folder).mkdir(parents=True)
        (stage/doc_folder/'WEIGHTS_MANIFEST.json').write_bytes(manifest_bytes)
        for name in DOCUMENTS:
            with open_asset(name) as stream:
                text = stream.read(1_000_001)
            if not text or len(text) > 1_000_000:
                raise ValueError('Invalid release document: '+name)
            text.decode('utf-8')
            (stage/doc_folder/name).write_bytes(text)
        # All downloads and checks finish before any destination files are installed.
        for relative in planned:
            path = destination/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            os.link(stage/relative, path)  # Exclusive creation; never replaces existing assets.
    return dict(status='installed', variant=variant, weight_files=len(selected),
                bytes=sum(row['bytes'] for row in selected), destination=str(destination))


class GitHubError(RuntimeError):
    def __init__(self, status, message):
        super().__init__('GitHub HTTP '+str(status)+': '+message)
        self.status = status


class GitHub:
    def __init__(self, token):
        self.token = token

    def request(self, method, url, payload=None, file=None):
        parts = urlsplit(url)
        if (parts.scheme != 'https' or parts.netloc not in ('api.github.com', 'uploads.github.com')
                or parts.username or parts.password):
            raise ValueError('Unexpected GitHub API endpoint')
        headers = {'Accept': 'application/vnd.github+json', 'User-Agent': 'qwen-klein-baseline',
                   'Authorization': 'Bearer '+self.token, 'X-GitHub-Api-Version': '2026-03-10'}
        body = json.dumps(payload).encode() if payload is not None else None
        if body is not None:
            headers['Content-Type'] = 'application/json'
        if file is not None:
            headers['Content-Type'] = 'application/octet-stream'
            headers['Content-Length'] = str(Path(file).stat().st_size)
            body = Path(file).open('rb')
        connection = http.client.HTTPSConnection(parts.netloc, timeout=60)
        try:
            connection.request(method, parts.path+('?' + parts.query if parts.query else ''),
                               body=body, headers=headers)
            response = connection.getresponse()
            content = response.read(2_000_001)
            try:
                data = json.loads(content) if content else {}
            except ValueError:
                raise GitHubError(response.status, 'Response was not JSON') from None
            if not 200 <= response.status < 300:
                raise GitHubError(response.status, data.get('message', 'Request failed'))
            return data
        finally:
            connection.close()
            if file is not None:
                body.close()


def publish_weights(directory, client):
    directory = Path(directory)
    manifest = validate_directory(directory)
    api = 'https://api.github.com/repos/'+manifest['repository']
    info = client.request('GET', api)
    if not info.get('permissions', {}).get('push'):
        raise PermissionError('Token needs Contents: Read and write for '+manifest['repository'])
    if info.get('private', True):
        raise ValueError('The repository must already be public for an open weight release')
    try:
        release = client.request('GET', api+'/releases/tags/'+quote(manifest['tag'], safe=''))
    except GitHubError as error:
        if error.status != 404:
            raise
        release = None
    # Drafts can be absent from the tag endpoint. Look for them before creating one.
    if release is None:
        releases = client.request('GET', api+'/releases?per_page=100')
        release = next((item for item in releases if item['tag_name'] == manifest['tag']), None)
    if release is not None and not release['draft']:
        raise FileExistsError('This release is already public; it will not be modified: '+release['html_url'])
    if release is None:
        release = client.request('POST', api+'/releases', payload={
            'tag_name': manifest['tag'], 'target_commitish': info['default_branch'],
            'name': 'Frozen contract and history baseline weights', 'draft': True,
            'body': (directory/'RELEASE_NOTES.md').read_text(), 'prerelease': False})
    release_api = api+'/releases/'+str(release['id'])
    names = [row['filename'] for row in manifest['weights']]+['WEIGHTS_MANIFEST.json', *DOCUMENTS]
    assets = client.request('GET', release_api+'/assets?per_page=100')
    existing = {row['name']: row for row in assets}
    if set(existing)-set(names):
        raise ValueError('Draft contains unexpected attachments; review it before continuing')
    for name in names:
        path = directory/name
        if name in existing:
            item = existing[name]
            if item['size'] != path.stat().st_size or item['state'] != 'uploaded':
                raise ValueError('Draft attachment conflicts or is incomplete: '+name)
            print('Already uploaded (name/size checked): '+name, flush=True)
            continue
        print('Uploading '+name, flush=True)
        upload = release['upload_url'].split('{', 1)[0]+'?'+urlencode({'name': name})
        item = client.request('POST', upload, file=path)
        if item['name'] != name or item['size'] != path.stat().st_size or item['state'] != 'uploaded':
            raise ValueError('Upload not complete: '+name)
    assets = client.request('GET', release_api+'/assets?per_page=100')
    actual = {item['name']: item['size'] for item in assets if item['state'] == 'uploaded'}
    if actual != {name: (directory/name).stat().st_size for name in names}:
        raise ValueError('Attachment list is incomplete; release remains a draft')
    published = client.request('PATCH', release_api, payload={
        'draft': False, 'body': (directory/'RELEASE_NOTES.md').read_text()})
    return dict(status='published', url=published['html_url'], assets=len(names))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export', help='Copy frozen project-trained weights to a new release directory')
    export.add_argument('--source', required=True, type=Path)
    export.add_argument('--output', required=True, type=Path)
    export.add_argument('--repo', default=REPOSITORY)
    export.add_argument('--tag', default=TAG)
    install = commands.add_parser('install', help='Download and install weights without overwriting existing files')
    install.add_argument('--output', type=Path, default=ROOT/'assets/QwenAnimeBaseline')
    install.add_argument('--variant', choices=(*VARIANTS, 'all'), default='all')
    install.add_argument('--from-directory', type=Path)
    install.add_argument('--repo', default=REPOSITORY)
    install.add_argument('--tag', default=TAG)
    publish = commands.add_parser('publish', help='Upload attachments to a draft, then publish the complete release')
    publish.add_argument('--directory', type=Path, required=True)
    publish.add_argument('--check-only', action='store_true', help='Check local files; no network or authentication')
    args = parser.parse_args()
    if args.command == 'export':
        manifest = export_weights(args.source, args.output, args.repo, args.tag)
        result = dict(status='exported', directory=str(args.output), weight_files=len(manifest['weights']),
                      bytes=sum(row['bytes'] for row in manifest['weights']))
    elif args.command == 'install':
        result = install_weights(args.output, args.variant, args.from_directory, args.repo, args.tag)
    else:
        manifest = validate_directory(args.directory)
        if args.check_only:
            result = dict(status='passed', weight_files=len(manifest['weights']), network_used=False)
        else:
            token = os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
            if not token:
                if not sys.stdin.isatty():
                    raise RuntimeError('Run in a terminal to enter a GitHub token, or set GH_TOKEN')
                token = getpass.getpass('GitHub token (Contents: Read and write; input hidden): ').strip()
            if not token:
                raise RuntimeError('No GitHub token supplied')
            result = publish_weights(args.directory, GitHub(token))
    print(json_text(result), end='')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(str(error))
