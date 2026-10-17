"""Build an inference-only D1 from C0 LoRA and the selected B fusion checkpoint."""
import argparse
import json
from pathlib import Path
import shutil


def compose(lora, fusion, output):
    sources = [Path(lora).resolve(), Path(fusion).resolve()]
    configs = [json.loads((p/'config.json').read_text()) for p in sources]
    for key in ('model', 'qwen', 'rank'):
        if configs[0][key] != configs[1][key]:
            raise ValueError('Incompatible component '+key)
    for path, name in zip(sources, ('adapter_model.safetensors', 'character_fusion.safetensors')):
        if not (path/'complete.json').is_file() or not (path/name).is_file():
            raise FileNotFoundError('Incomplete source checkpoint: '+str(path))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    components = {}
    for role, path, config, name in zip(('lora', 'fusion'), sources, configs,
                                      ('adapter_model.safetensors', 'character_fusion.safetensors')):
        shutil.copyfile(path/name, output/name)
        components[role] = {'checkpoint': str(path), 'file': str(path/name),
                            'training_step': config['step'], 'bytes': (output/name).stat().st_size}
    config = {key:configs[0][key] for key in ('model', 'qwen', 'data', 'rank', 'combined_parameters', 'native_sequence_length')}
    config.update(stage='character_inference_composite', protocol='character_component_cross_v1',
                  step=None, inference_only=True, training_resume_supported=False,
                  component_sources=components, one_component_per_role=True)
    (output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    (output/'complete.json').write_text(json.dumps({'artifact_type':'inference_component_combination',
        'component_sources':components, 'files':['adapter_model.safetensors','character_fusion.safetensors','config.json']},indent=2)+'\n')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lora', type=Path, required=True)
    parser.add_argument('--fusion', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(compose(args.lora, args.fusion, args.output))


if __name__ == '__main__':
    main()
