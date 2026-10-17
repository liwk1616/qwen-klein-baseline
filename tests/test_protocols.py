"""CPU regression checks for protocol boundaries and portable handoff behavior."""
import contextlib
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'core'))
from asset_paths import configure, relocate
from independent_candidate_protocol import make_request, select_candidate, candidate_rank, SEED_STRIDE
from storygptv_protocol import causal_history
from causal_candidate_protocol import parse_grade_response
from run_metrics import configured_stories, summarize, VARIANTS
import run_pilot
from tools.compose_d1 import compose


def story(identifier='story', split='test'):
    return {'story_id': identifier, 'task_id': identifier, 'split': split, 'initial_original_images': [],
            'targets': [{'clip_id': f'{identifier}_{i}', 'example_id': f'{identifier}_frame{i+1}',
                         'caption': 'Fred stands in the kitchen.', 'seed': 100+i,
                         'position': i, 'split': split,
                         'initial_reference_bank': [{'name':'fred','image':'/bank/fred.jpeg'}] if i == 0 else []}
                        for i in range(4)]}


class ProtocolTests(unittest.TestCase):
    def test_only_requested_variants_are_exposed(self):
        self.assertEqual(set(VARIANTS), {'raw_single', 'raw_best4', 'history_best4'})

    def test_raw_uses_current_caption_without_history(self):
        source = story()
        request = make_request(0, source, 2, 'Wilma is outside.', {'wilma':'/bank/wilma.jpeg'}, 'raw', 4)
        self.assertEqual(request['target_caption'], 'Wilma is outside.')
        self.assertEqual(request['context'], {'history': []})
        self.assertEqual(request['references'], [{'name':'wilma','image':'/bank/wilma.jpeg'}])
        self.assertEqual(request['seeds'], [102+k*SEED_STRIDE for k in range(4)])

    def test_original_subset_seed_rule_survives_limiting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stories = [story('a'), story('b'), story('c')]
            (root/'all.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in stories))
            (root/'subset.jsonl').write_text(json.dumps(stories[1])+'\n')
            config = {'raw_stories':str(root/'all.jsonl'), 'raw_seed_subset':str(root/'subset.jsonl')}
            full = configured_stories(config, 'raw_best4')[1]
            limited = configured_stories(config, 'raw_best4', limit=2)[1]
            self.assertEqual(full[:2], limited)
            self.assertEqual(full[1]['targets'][0]['seed'], 20260916)
            self.assertEqual(full[2]['targets'][3]['seed'], 20260916+2*4+3)

    def test_wrong_split_and_ground_truth_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'stories.jsonl'
            for change in [{'split':'test'}, {'initial_original_images':['target.jpg']}]:
                value = {**story(split='val'), **change}
                path.write_text(json.dumps(value)+'\n')
                with self.assertRaises(ValueError):
                    configured_stories({'stories':str(path)}, 'history_best4')

    def test_ties_choose_lowest_candidate_and_failures_remain(self):
        request = make_request(0, story(), 0, 'Fred stands.', {'fred':'/bank/fred.jpeg'}, 'raw', 4)
        candidates = [dict(candidate_index=i, example_id=f'candidate{i}', base_example_id=request['example_id'],
                           frame_index=0, seed=seed, status='complete') for i,seed in enumerate(request['seeds'])]
        classifier = {v['example_id']:{'character':.5,'background':.5} for v in candidates}
        visual = {v['example_id']:{'scores':None,'error':'invalid JSON'} for v in candidates}
        chosen, ranks = select_candidate(request, candidates, classifier, visual)
        self.assertEqual(chosen, 0)
        self.assertEqual(len(ranks), 4)
        self.assertTrue(all(r['components']['visual'] == .5 for r in ranks))
        with self.assertRaises(ValueError):
            select_candidate(request, candidates[:-1], classifier, visual)
        candidates[0]['seed'] += 1
        with self.assertRaises(ValueError):
            select_candidate(request, candidates, classifier, visual)

    def test_ranking_weights_and_invalid_grades(self):
        score, _ = candidate_rank({'character':1.,'background':0.}, {'scores':None,'error':'failed'})
        self.assertAlmostEqual(score, .45+.30*.5)
        with self.assertRaises(ValueError):
            candidate_rank({'character':float('nan'),'background':0.}, {'scores':None,'error':'failed'})

    def test_history_grader_preserves_frozen_case_sensitive_parser(self):
        response = 'visible: Fred stands.\nidentity: 4\ncast: 4\nbackground: 3\naction: 3\nquality: 4'
        parse_grade_response(response)
        with self.assertRaises(ValueError):
            parse_grade_response(response.replace('visible:', 'Visible:'))

    def test_history_requires_exact_generated_prefix(self):
        value = story()
        selected = {'story_id':'story','clip_id':'story_0','image':'generated.bmp',
                    'status':'complete','kind':'generated_story_frame'}
        result = causal_history('story', value['targets'], [selected], 1)
        self.assertEqual([h['image'] for h in result['history']], ['generated.bmp'])
        for change in [{'kind':'ground_truth'}, {'story_id':'another'}, {'clip_id':'story_1'}]:
            with self.assertRaises(ValueError):
                causal_history('story', value['targets'], [{**selected, **change}], 1)
        with self.assertRaises(ValueError):
            causal_history('story', value['targets'], [selected], 2)

    def test_missing_output_with_empty_labels_is_incorrect(self):
        result = summarize([{'char_target_indices':[], 'char_predicted_indices':None},
                            {'char_target_indices':[], 'char_predicted_indices':[]}], 'char')
        self.assertEqual(result['frames'], 2)
        self.assertEqual(result['correct'], 1)
        self.assertEqual(result['accuracy_percent'], 50.)

    def test_relocation_is_recursive_and_respects_directory_boundaries(self):
        with patch.dict(os.environ):
            configure([('/old/project', '/new/location')])
            actual = relocate({'path':'/old/project/models/x', 'list':['/old/project/data/x'],
                               'caption':'Fred sees /old/project/data/x', 'other':'/old/project_other/x'})
            self.assertEqual(actual['path'], '/new/location/models/x')
            self.assertEqual(actual['list'], ['/new/location/data/x'])
            self.assertEqual(actual['caption'], 'Fred sees /old/project/data/x')
            self.assertEqual(actual['other'], '/old/project_other/x')

    def test_pilot_shares_first_frame_and_keeps_independent_branches(self):
        case = {'case_id':'01','case_index':0,'story_id':'story_0','arms':{}}
        for arm,count in zip(run_pilot.ARMS, (5,5,4)):
            targets = story()['targets']
            if count == 5:
                targets.append({**targets[-1], 'clip_id':arm+'_4', 'position':4,'seed':104})
            case['arms'][arm] = targets
        run_pilot.validate_cases([case])
        captured = []
        def generate(model, root, request):
            captured.append(request)
            return {'story_id':request['story_id'], 'clip_id':request['clip_id'],
                    'image':request['example_id']+'.bmp', 'status':'complete','kind':'generated_story_frame'}, True
        fake = types.ModuleType('models')
        fake.Models = lambda plan: types.SimpleNamespace(torch=types.SimpleNamespace(inference_mode=contextlib.nullcontext))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, models=fake), patch.dict(os.environ, CUDA_VISIBLE_DEVICES='0'):
            root = Path(tmp)
            (root/'inputs').mkdir()
            (root/'inputs/cases.json').write_text(json.dumps([case]))
            (root/'run_plan.json').write_text(json.dumps({'gpus':[0], 'path_mappings':[]}))
            with patch.object(run_pilot, 'generate_frame', generate):
                run_pilot.worker(root, 0)
            sequences = [json.loads(p.read_text()) for p in sorted((root/'frames').glob('*_sequence.json'))]
            self.assertEqual(len(captured), 12)
            self.assertEqual(sum(len(v) for v in sequences), 14)
            self.assertEqual(len({s[0]['image'] for s in sequences}), 1)
            for request in captured:
                history = request['context']['history']
                self.assertEqual(len(history), request['target_position']-1)
                if len(history) > 1:
                    arm = request['task_id'].split('pilot20_01_',1)[1]
                    self.assertTrue(all(arm in h['image'] for h in history[1:]))

    def test_d1_combines_correct_roles_and_is_not_resumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name,step in [('c0',25000), ('b',26000)]:
                path = root/name
                path.mkdir()
                (path/'config.json').write_text(json.dumps(dict(model='m',qwen='q',data='d',rank=32,
                    combined_parameters=1,native_sequence_length=128,step=step)))
                (path/'complete.json').write_text('{}')
                (path/'adapter_model.safetensors').write_bytes((name+' lora').encode())
                (path/'character_fusion.safetensors').write_bytes((name+' fusion').encode())
            output = compose(root/'c0', root/'b', root/'d1')
            self.assertEqual((output/'adapter_model.safetensors').read_bytes(), b'c0 lora')
            self.assertEqual((output/'character_fusion.safetensors').read_bytes(), b'b fusion')
            self.assertFalse(json.loads((output/'config.json').read_text())['training_resume_supported'])
            with self.assertRaises(FileExistsError):
                compose(root/'c0', root/'b', root/'d1')


if __name__ == '__main__':
    unittest.main()
