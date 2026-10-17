# 模型与数据交接

本仓库只保存代码、配置模板和小型历史指标摘要。两条 baseline 共用基础模型、Qwen、固定图库和评分器，使用不同的人物权重。完整资源保持原项目的相对目录布局，路径不依赖原服务器挂载位置。

## 资源导出

在已有原项目的机器上，先列出推理资源；此命令只输出清单，不复制文件：

```bash
python tools/export_assets.py --source /path/to/QwenAnimeBaseline --pilot-cases /path/to/pilot/run_001/inputs/cases.json
```

需要转交到另一块磁盘时，指定一个尚不存在的目标目录：

```bash
python tools/export_assets.py --source /path/to/QwenAnimeBaseline \
  --pilot-cases /path/to/pilot/run_001/inputs/cases.json --copy-to /path/to/baseline-assets
```

目标布局为：

```text
baseline-assets/
  QwenAnimeBaseline/
    models/...
    data/...
    outputs/...       # 此处保存冻结 checkpoint 和图库特征，属于运行输入
  pilot20/cases.json
  ASSET_INVENTORY.json
```

在接收端将 `project_root` 指向 `baseline-assets/QwenAnimeBaseline`，`pilot_cases` 指向 `baseline-assets/pilot20/cases.json`。或者把该目录的内容放入仓库的 `assets/`，直接使用示例配置。

导出器按显式文件列表复制，解引用 Klein Base 的 `text_encoder`、`tokenizer` 和 `vae` 符号链接，使接收端不再依赖原机链接目标。它不复制模型下载缓存，不计算哈希或校验和。历史文件里的路径由运行时映射到新根目录，原始配置和 safetensors 元数据保持不变。

本机清单为 59 个推理资源文件、26,107,422,455 字节，约 24.31 GiB；可选 pilot 输入另计。配置文件里 `complete.json` 的完整训练文件列表可能包含优化器，而推理只需要下面列出的推理文件。

## 推理必需资源

下列路径均相对于 `project_root`：

| 资源 | 相对路径或文件 |
| --- | --- |
| 非蒸馏 Klein Base | `models/flux2-klein-base-4b/` 的 model_index、scheduler、transformer、text_encoder、tokenizer、vae |
| Qwen3.5 4B | `models/qwen3.5-4b/` 的权重、索引、配置、tokenizer、processor、chat template |
| raw D1 | `outputs/flintstones_klein_component_cross_validation_20260920/run_001/components/D1/` |
| history C0 25k | `outputs/flintstones_klein_four_gpu_20260918/run_001/character_training_025000/checkpoint_025000/` |
| history mapper 8k | `outputs/flintstones_klein_story_continue_20260919/run_001/story_training_008000/checkpoint_008000/` |
| 固定图库特征 | `outputs/flintstones_klein_aligned_fresh_20260916/run_001/character_training_000000/official_character_features.safetensors` |
| 七人固定图库图片 | `outputs/flintstones_klein_storygptv_protocol_20260916/audit_001/official_reference_bank/` |
| 图库顺序清单 | `data/flintstones_storygptv_20260916/released_001/reference_bank.json` |
| raw 完整 test 与种子子集 | 同一 released_001 下的 `evaluation/test_generation_stories.jsonl`、`test10_existing_generation_stories.jsonl` |
| 原始 caption | `data/flintstones/cleaned_annotations.json` |
| history 开发组 | `data/flintstones_validation256_20261002/v1/inputs/stories_256.jsonl` |
| 背景标签顺序 | 同一 inputs 下的 `background_names.json` |
| 评价标签 | `data/flintstones/labels.pkl`、`labels_bg.pkl` |
| 评价器 | `data/flintstones/evaluators_safetensors_20260915/classifier_char.safetensors`、`classifier_bg.safetensors` |

人物 checkpoint 需要 `config.json`、`complete.json`、`adapter_model.safetensors`、`character_fusion.safetensors`；mapper 需要 `config.json`、`complete.json`、`mapper.safetensors`。不能用蒸馏 Klein、D1 融合层或其他 mapper 替换 history 的配对权重。

图像生成不读取目标真实帧，评分只读取标签，因此推理导出包不需要全部视频帧和分割掩码。标签在生成和选图之后用于最终评价。

## 训练补充资源

导出时加 `--training`，额外包含：released_001 的 `clips/`、`stories/`、`protocol.json`、`teacher_conditioning.jsonl`；`data/flintstones/video_frames_sampled_4x/`；`data/downloads/flintstones.zip`；完整 C0 25k checkpoint（优化器、RNG、历史覆盖记录）。原始视频帧与掩码体积较大，先不带 `--copy-to` 查看清单和总量。

8k mapper 的现有目录没有 `optimizer.safetensors`。可以用它推理，不能把它视作完整的续训检查点。D1 是组合推理产物，也不能作为训练恢复点。

模型与数据来源、原项目资源定位均保留在清单和配置中。对外提供第三方权重或数据时沿用其各自的许可与获取方式，代码仓库不附带这些资产。
