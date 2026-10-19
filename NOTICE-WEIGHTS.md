# Weight license scope and attribution

The five project-trained inference checkpoint files listed below are made available
under the Apache License, Version 2.0, included in `LICENSE-WEIGHTS`:

- `raw-d1-lora.safetensors`
- `raw-d1-fusion.safetensors`
- `history-c0-lora.safetensors`
- `history-c0-fusion.safetensors`
- `history-mapper-8k.safetensors`

This license grant covers the project contributions to these checkpoints. It does
not relicense the source code inherited from other projects, training data,
reference images, or third-party evaluator checkpoints. See `THIRD_PARTY.md` in
the source repository for those origins.

These checkpoints adapt FLUX.2 [klein] Base 4B by Black Forest Labs and use
Qwen3.5 4B by the Qwen team. Both official model repositories identify their
weights as Apache-2.0:

- https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B
- https://huggingface.co/Qwen/Qwen3.5-4B

The release contains project-trained LoRA, character fusion, and story mapper
weights, rather than a copy of either foundation model. The LoRA adapts the Klein
transformer; the fusion and mapper modules implement this project's conditioning
interfaces. In particular, the raw D1 fusion comes from reference training step
26000, while the history fusion comes from character training step 25000.
The history mapper was trained to step 8000 against that C0 character checkpoint.

Flintstones benchmark data, reference images, and evaluator models originate from
the StoryGPT-V resource distribution. Their original authors and rights holders
retain their rights. Access them through their upstream sources:
https://github.com/xiaoqian-shen/StoryGPT-V

This is an independent project release; the upstream model and dataset authors
have not endorsed it. Model outputs are not a grant of rights to third-party
characters or other protected material.
