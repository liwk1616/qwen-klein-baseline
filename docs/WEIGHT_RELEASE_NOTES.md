# Frozen contract and history baseline weights

公开两条原始 baseline 的项目训练推理权重，保持现有冻结版本。

| 版本 | 必需的权重附件 |
| --- | --- |
| 合同指标版 `raw_best4` / `raw_single` | `raw-d1-lora.safetensors`、`raw-d1-fusion.safetensors` |
| 历史版 `history_best4` / `pilot` | `history-c0-lora.safetensors`、`history-c0-fusion.safetensors`、`history-mapper-8k.safetensors` |

五个权重文件合计约 454.80 MiB。D1 使用 C0 25k LoRA＋B 26k 融合层；历史版
使用 C0 25k LoRA＋C0 25k 融合层＋配对的 8k mapper。请勿混用融合层或 mapper。

在本仓库最新代码中运行：

```bash
python tools/weights.py install --variant all --output assets/QwenAnimeBaseline
```

下载脚本按照 `WEIGHTS_MANIFEST.json` 安装权重和原始 checkpoint 配置，
不覆盖已有文件。也可按版本选择 `--variant raw_best4` 或 `--variant history_best4`。

这次附件只包含项目训练出的增量权重。基础模型、固定图库、数据和评价器仍需单独准备：

- [FLUX.2 Klein Base 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)
- [Qwen3.5 4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- [模型卡](https://github.com/liwk1616/qwen-klein-baseline/blob/main/docs/MODEL_CARD.md)
- [完整资源交接清单](https://github.com/liwk1616/qwen-klein-baseline/blob/main/docs/ASSETS.md)

项目权重采用 Apache-2.0；参见附件 `LICENSE-WEIGHTS` 和 `NOTICE-WEIGHTS.md`。
本发布是推理权重，不是完整续训快照。历史成绩见模型卡；打包时没有新训练或重新测分。
