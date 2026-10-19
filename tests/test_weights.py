"""Release safety, frozen pairing, and offline installation regression tests."""
from copy import deepcopy
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import weights


def tiny_weight(path, value):
    header = json.dumps({'weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
    header += b' ' * (-len(header) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack('<Q', len(header))+header+struct.pack('<f', value))


class FakeGitHub:
    def __init__(self, fail_upload=None, public=True, push=True):
        self.fail_upload, self.public, self.push = fail_upload, public, push
        self.assets, self.calls = [], []

    def request(self, method, url, payload=None, file=None):
        self.calls.append((method, url, payload))
        if file is not None:
            if len(self.assets) == self.fail_upload:
                raise weights.GitHubError(503, 'Test upload interruption')
            row = dict(name=file.name, size=file.stat().st_size, state='uploaded')
            self.assets.append(row)
            return row
        if url.endswith('/assets?per_page=100'):
            return deepcopy(self.assets)
        if '/releases/tags/' in url:
            raise weights.GitHubError(404, 'Not Found')
        if url.endswith('/releases?per_page=100'):
            return []
        if method == 'POST':
            return dict(id=1, draft=True, upload_url='https://uploads.github.com/repos/example/model/releases/1/assets{?name,label}')
        if method == 'PATCH':
            return dict(html_url='https://github.com/example/model/releases/tag/weights-v1')
        return dict(private=not self.public, permissions=dict(push=self.push), default_branch='main')


class WeightReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source, self.release = self.root/'source', self.root/'release'
        config = weights.settings()
        recorded = '/frozen-assets/QwenAnimeBaseline'
        for index, spec in enumerate(weights.weight_specs()):
            tiny_weight(self.source/spec['path'], index+1)
        for spec in weights.metadata_specs():
            path = self.source/spec['path']
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            if path.name == 'config.json':
                data = dict(model=recorded+'/'+config['model'], step=25000)
                if spec['path'].startswith(config['mapper_checkpoint']+'/'):
                    data.update(step=8000, character_checkpoint=recorded+'/'+config['character_checkpoint'])
            path.write_text(json.dumps(data))
        self.manifest = weights.export_weights(self.source, self.release)

    def test_export_and_install_keep_bytes_and_roles(self):
        output = self.root/'installed'
        with patch('sys.stdout', new_callable=io.StringIO):
            result = weights.install_weights(output, source_dir=self.release)
        self.assertEqual(result['weight_files'], 5)
        for spec in weights.weight_specs()+weights.metadata_specs():
            self.assertEqual((output/spec['path']).read_bytes(), (self.source/spec['path']).read_bytes())
        self.assertNotEqual((output/weights.weight_specs()[1]['path']).read_bytes(),
                            (output/weights.weight_specs()[3]['path']).read_bytes())

    def test_variant_download_does_not_install_other_baseline(self):
        output = self.root/'raw'
        with patch('sys.stdout', new_callable=io.StringIO):
            result = weights.install_weights(output, variant='raw_best4', source_dir=self.release)
        self.assertEqual(result['weight_files'], 2)
        for spec in weights.weight_specs():
            self.assertEqual((output/spec['path']).exists(), spec['variant'] == 'raw_best4')

    def test_existing_assets_are_never_overwritten(self):
        output = self.root/'installed'
        protected = output/weights.weight_specs()[0]['path']
        protected.parent.mkdir(parents=True)
        protected.write_bytes(b'Existing user checkpoint')
        with self.assertRaises(FileExistsError):
            weights.install_weights(output, source_dir=self.release)
        self.assertEqual(protected.read_bytes(), b'Existing user checkpoint')
        self.assertFalse((output/weights.weight_specs()[1]['path']).exists())

    def test_incomplete_download_does_not_install_partial_weights(self):
        target = self.release/weights.weight_specs()[1]['filename']
        target.write_bytes(target.read_bytes()[:-1])
        output = self.root/'installed'
        with patch('sys.stdout', new_callable=io.StringIO), self.assertRaises(ValueError):
            weights.install_weights(output, source_dir=self.release)
        self.assertFalse(output.exists())

    def test_manifest_rejects_path_escape_and_wrong_mapper_pairing(self):
        broken = deepcopy(self.manifest)
        broken['weights'][0]['path'] = '../unexpected'
        with self.assertRaises(ValueError):
            weights.validate_manifest(broken)
        broken = deepcopy(self.manifest)
        spec = weights.settings()['mapper_checkpoint']+'/config.json'
        row = next(item for item in broken['metadata'] if item['path'] == spec)
        value = json.loads(row['text'])
        value['character_checkpoint'] = '/different/checkpoint'
        row['text'] = json.dumps(value)
        row['bytes'] = len(row['text'].encode())
        with self.assertRaises(ValueError):
            weights.validate_manifest(broken)

    def test_upload_failure_leaves_draft_unpublished(self):
        client = FakeGitHub(fail_upload=1)
        with patch('sys.stdout', new_callable=io.StringIO), self.assertRaises(weights.GitHubError):
            weights.publish_weights(self.release, client)
        created = [payload for method, url, payload in client.calls if method == 'POST' and payload]
        self.assertTrue(created[0]['draft'])
        self.assertFalse(any(method == 'PATCH' for method, _, _ in client.calls))

    def test_all_attachments_must_upload_before_publication(self):
        client = FakeGitHub()
        with patch('sys.stdout', new_callable=io.StringIO):
            result = weights.publish_weights(self.release, client)
        self.assertEqual(result['assets'], 9)
        self.assertEqual(len(client.assets), 9)
        self.assertEqual(client.calls[-1][0], 'PATCH')
        self.assertFalse(client.calls[-1][2]['draft'])

    def test_missing_write_access_and_private_repo_stop_before_creation(self):
        for client, error in ((FakeGitHub(push=False), PermissionError),
                              (FakeGitHub(public=False), ValueError)):
            with self.assertRaises(error):
                weights.publish_weights(self.release, client)
            self.assertFalse(any(method in ('POST', 'PATCH') for method, _, _ in client.calls))

    def test_authenticated_upload_never_follows_untrusted_endpoint(self):
        with self.assertRaises(ValueError):
            weights.GitHub('test-credential').request('POST', 'https://example.com/upload')


if __name__ == '__main__':
    unittest.main()
