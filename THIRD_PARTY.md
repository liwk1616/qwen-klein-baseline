# 来源说明

本包整理自既有 QwenAnimeBaseline 与 kuadu 项目，逐文件来源见 `SOURCE_MAP.json`。运行依赖 PyTorch、torchvision、Transformers、Diffusers、PEFT、safetensors、NumPy、Pillow、Accelerate 等第三方组件；依赖由 requirements 安装，不随仓库复制环境。

`core/public_transforms.py` 继承原项目审阅并保存的 StoryGPT-V 图像变换代码，来源快照是 `QwenAnimeBaseline/outputs/flintstones_klein_reference_alignment_20260920/run_001/source/public_transforms.py`。Flintstones 发布清单、七人图库、人物与背景评价器沿用原项目中的 StoryGPT-V 资源记录。

Qwen3.5 4B、FLUX.2 Klein Base 4B 及其 tokenizer、VAE、text encoder 作为外部资产提供；导出器保留所复制模型目录中的 LICENSE 文件（若存在）。本文件是来源清单，不替代这些组件的许可证，也不授予模型或数据新的再分发权限。

项目自行训练的五个推理权重文件按 [LICENSE-WEIGHTS](LICENSE-WEIGHTS) 的 Apache-2.0 条款提供，适用文件和来源见 [NOTICE-WEIGHTS.md](NOTICE-WEIGHTS.md)。这项权重授权不统一改写继承代码或第三方数据、图片、分类器的许可证。

官方模型和资源入口：

- [FLUX.2 Klein Base 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)：官方标注 Apache-2.0，必须使用非蒸馏 Base 版本。
- [Qwen3.5 4B](https://huggingface.co/Qwen/Qwen3.5-4B)：官方标注 Apache-2.0。
- [StoryGPT-V](https://github.com/xiaoqian-shen/StoryGPT-V)：Flintstones 数据、参考图库和分类器的来源；公开项目权重附件不包含这些第三方资产。
