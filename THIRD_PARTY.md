# 来源说明

本包整理自既有 QwenAnimeBaseline 与 kuadu 项目，逐文件来源见 `SOURCE_MAP.json`。运行依赖 PyTorch、torchvision、Transformers、Diffusers、PEFT、safetensors、NumPy、Pillow、Accelerate 等第三方组件；依赖由 requirements 安装，不随仓库复制环境。

`core/public_transforms.py` 继承原项目审阅并保存的 StoryGPT-V 图像变换代码，来源快照是 `QwenAnimeBaseline/outputs/flintstones_klein_reference_alignment_20260920/run_001/source/public_transforms.py`。Flintstones 发布清单、七人图库、人物与背景评价器沿用原项目中的 StoryGPT-V 资源记录。

Qwen3.5 4B、FLUX.2 Klein Base 4B 及其 tokenizer、VAE、text encoder 作为外部资产提供；导出器保留所复制模型目录中的 LICENSE 文件（若存在）。本文件是来源清单，不替代这些组件的许可证，也不授予模型或数据新的再分发权限。
