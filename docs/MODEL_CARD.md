# Frozen Qwen Klein baseline weights

本模型卡对应 `liwk1616/qwen-klein-baseline` 的 `weights-v1` 发布文件。
发布位置：https://github.com/liwk1616/qwen-klein-baseline/releases

这是两条冻结 baseline 的推理权重。训练入口、协议、环境和数据要求见
https://github.com/liwk1616/qwen-klein-baseline 。五个文件共 476,893,264 字节
（约 454.80 MiB），不包含优化器或完整续训状态。

## 文件与版本

| 文件 | 字节数 | 用途 |
| --- | ---: | --- |
| `raw-d1-lora.safetensors` | 149442256 | 合同指标版 D1：C0 25k LoRA |
| `raw-d1-fusion.safetensors` | 36815520 | 合同指标版 D1：B 26k 融合层 |
| `history-c0-lora.safetensors` | 149442256 | 历史版 C0 25k LoRA |
| `history-c0-fusion.safetensors` | 36815520 | 历史版 C0 25k 融合层 |
| `history-mapper-8k.safetensors` | 104377712 | 历史版 8k mapper；必须配对 C0 25k |

`raw_single` 与 `raw_best4` 共用 D1，区别为候选数与选图协议。两条版本的
LoRA 均来自 C0 25k；分别命名是为了让下载和加载时的版本边界清楚。
两个融合层即使文件长度相同，也不能互换。推理权重张量和原始文件字节不改写。

`WEIGHTS_MANIFEST.json` 给出每个附件的安装路径、文件长度、显式张量布局和
原始 checkpoint JSON。历史 JSON 中出现的旧机器路径用于来源记录，代码会
映射到使用者配置的 `project_root`；接收者不需要建立那些旧路径。
原始 `complete.json` 还可能列出训练恢复文件；本发布清单才是此次提供的文件范围。

## 基础模型和其他输入

- 非蒸馏基础模型：[black-forest-labs/FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)。包含 transformer、text encoder、tokenizer、VAE、scheduler。
- 理解与视觉编码模型：[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)。
- Flintstones 标注、人物图库与分类器：[StoryGPT-V 官方资源](https://github.com/xiaoqian-shen/StoryGPT-V)。
- 冻结故事划分、种子子集、图库特征及本项目使用的 safetensors 分类器转换件，仍按源码仓库 `docs/ASSETS.md` 的交接清单安装。

基础模型由官方单独提供；本发布的五个附件是项目训练出的增量权重。
只有这些附件还不能直接运行完整 benchmark。原项目完整推理资源约 24.31 GiB。
使用方需准备仓库中记录的固定输入清单；随意更换划分、图库或候选数不能复现历史分数。

## 下载与加载

在源码仓库中运行（发布完成后可用）：

```bash
python tools/weights.py install --variant all --output assets/QwenAnimeBaseline
```

只下载合同指标版时选择 `--variant raw_best4`；只下载历史版时选择
`--variant history_best4`。安装器使用标准库，不需要先安装 PyTorch；它不覆盖
已有文件。已有资产目录的使用者可跳过已具备的 checkpoint 安装。

准备其他资源和两套 Python 环境后：

```bash
cp configs/assets.example.json configs/assets.local.json
python run.py check --runtime
python run.py metrics prepare --variant raw_best4 --limit-stories 2 --gpus 0 --output outputs/raw_smoke
python run.py metrics run --output outputs/raw_smoke
```

历史版将上述 `raw_best4` 改成 `history_best4`，并使用新的输出目录。
这些 safetensors 是本项目加载器使用的训练模块文件；完整加载入口是仓库的
`run.py`，其中 mapper 与融合层由项目代码装配。

## 训练和评测范围

权重来自 2026-09-18 至 2026-09-20 的原始训练阶段。合同指标版采用原始当帧
caption、固定人物图库、无历史、四候选选图；历史版使用本分支已生成的历史、
冻结的自动指代解析、当帧图库和 mapper。没有加入后续 SDO、state transition、
routing gate、history response 改进。

| 冻结流程 | 数据范围 | Char Acc % | BG Acc % |
| --- | --- | ---: | ---: |
| raw single | 865 test 故事，3460 帧 | 74.97 | 52.08 |
| raw best4 | 865 test 故事，3460 帧 | 90.29 | 64.45 |
| history best4 | 64 开发 val 故事，256 帧 | 91.41 | 70.31 |

以上为原实验历史指标，发布打包没有重新运行 GPU 评测。不同流程的数据范围不同，
不能由表中差值推断同条件提升。best4 使用分类器参与选图；分类器指标不等同于
画面质量、动作正确性或故事连贯性。原记录的 `formal_contract_benchmark` 为 `false`。
这组权重针对 Flintstones；没有对一般人物、其他动画风格或新数据分布验证同等表现。

## 授权与验证

五个项目训练权重按 Apache-2.0 提供，完整条款见同批附件 `LICENSE-WEIGHTS`，
范围和来源见 `NOTICE-WEIGHTS.md`。代码及第三方资产的授权分别记录，不由此声明统一更改。

发布工具与安装器检查显式文件清单、大小、JSON、safetensors 张量名称、形状、
类型和数据区长度；不计算或比对内容哈希、摘要、校验和或内容指纹。
这些检查能够发现缺失、截断和布局不匹配，不构成对同长度张量内容的密码学认证。
