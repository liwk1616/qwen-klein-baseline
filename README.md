# Qwen Klein Baseline 交接仓库

本仓库保留两条已冻结的 Flintstones baseline，供接手、复现和后续对照实验使用。代码包含数据准备、推理、自动选图、指标计算和原有训练入口。项目训练权重通过 [GitHub Releases](https://github.com/liwk1616/qwen-klein-baseline/releases) 单独发布，按 Apache-2.0 提供；基础模型、数据集与生成结果通过外部资源目录提供。权重清单见 [模型卡](docs/MODEL_CARD.md)，发布和下载操作见 [权重说明](docs/WEIGHTS.md)。

| 版本 | 入口 | 输入与历史 | 人物权重 | Mapper |
| --- | --- | --- | --- | --- |
| 合同指标版 | `metrics --variant raw_best4` | 原始当帧 caption，逐帧独立生成，固定人物图库，四选一 | D1：C0 25k LoRA＋B 26k 融合层 | 关闭 |
| 原始历史续写版 | `metrics --variant history_best4`；`pilot` | 当前描述＋本分支已选中的生成历史，自动指代解析、当帧图库，四选一 | C0 25k LoRA＋C0 25k 融合层 | 8k |

历史续写版对应 `kuadu/baseline-pilot20-stride8/run_001` 使用的冻结流程，以及其来源 `automatic_named_bank_own_history_20261004_v1`。这里的“无新增改进”指以该冻结版本为界：保留当时已有的指代解析、图库绑定、mapper 和选图规则，不加入后续 SDO、state transition、routing gate、history response 或新增训练模块。代码中的 `progressive_protocol.py` 是该冻结流程已经依赖的解析与评分工具。

## 历史成绩

这些数值来自既有实验记录，本次整理未重新出图测分。

| 流程与数据范围 | 故事／帧数 | Char Acc % | BG Acc % | Char F1 % | BG F1 % |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw single，完整 test | 865／3460 | 74.97 | 52.08 | 87.95 | 59.36 |
| raw best4，完整 test | 865／3460 | 90.29 | 64.45 | 95.52 | 71.17 |
| history best4，开发 val | 64／256 | 91.41 | 70.31 | 96.27 | 76.76 |

raw best4 的完整预算为 13,840 张候选；history 开发组为 1,024 张候选。两行使用不同数据集和输入协议，不能将数字差值解释为同条件提升。官方分类器参与 best4 选图，最终 Char/BG 也是该分类器的标签集合指标；它们不等同于人物外观、动作或叙事质量评价。“合同指标版”是项目中的版本名称，原始记录的 `formal_contract_benchmark` 为 `false`。

原始汇总保存在 [results/](results/)，20-case 跨度结果和指标定义见 [协议说明](docs/PROTOCOLS.md)。

## 环境与资源

使用 Linux、Python 3.10 和支持 BF16 的 NVIDIA CUDA 环境。推理进程按 GPU 分片，每个分片装载完整模型；历史版还在同一进程装载 Qwen、Klein 和分类器，因此需要足够的单卡显存。本仓库没有验证最低显存门槛。

沿用已有运行环境的两个 transformers 版本，分别安装：

```bash
python3.10 -m venv .venv-qwen
.venv-qwen/bin/python -m pip install -r requirements-qwen.txt
python3.10 -m venv .venv-score
.venv-score/bin/python -m pip install -r requirements-flux.txt
cp configs/assets.example.json configs/assets.local.json
```

依赖文件记录已运行环境的顶层版本；本次未从零安装环境。PyTorch wheel 需与部署机器的 CUDA 环境匹配。两个环境分别使用 transformers 5.15.1 与 4.57.6。

将师兄提供的资源放入 `assets/`，或修改 `configs/assets.local.json` 指向已有资源目录。该配置中的相对路径以配置文件所在目录为基准；`configs/metrics.json` 中的资源路径以 `project_root` 为基准。也可用 `python run.py --assets /path/to/assets.json ...` 指定外部本机配置。

资源清单、复制命令以及模型符号链接处理见 [资源交接](docs/ASSETS.md)。两个 baseline 的完整推理资源合计约 24.31 GiB，其中本项目训练的五个权重文件约 454.80 MiB。Release 发布完成后，可以自动下载这部分权重并恢复目录结构：

```bash
python tools/weights.py install --variant all --output assets/QwenAnimeBaseline
```

此命令只安装本项目 checkpoint；基础模型、图库、评分器和评测数据仍按资源清单准备。已有原项目权重目录的使用者可直接配置路径，无需重复安装。

先运行 CPU 检查：

```bash
python run.py check --source-only
python -B -m unittest discover -s tests -v
python run.py check --runtime
```

最后一条检查模型文件、配对 checkpoint、图库、全量输入及本地 tokenizer，不加载完整模型做推理。训练资源另用 `python run.py check --training` 检查。

## 合同指标版

先准备两个故事的最小任务，再执行 GPU 推理。`prepare` 不出图，`run` 才会启动 GPU 工作；以下输出目录必须尚不存在。

```bash
python run.py metrics prepare --variant raw_best4 --limit-stories 2 --gpus 0 --output outputs/raw_smoke
python run.py metrics run --output outputs/raw_smoke
```

完整 test：

```bash
python run.py metrics prepare --variant raw_best4 --gpus 0,1,2,3 --output outputs/raw_full
python run.py metrics run --output outputs/raw_full
```

每帧保留四张 BMP 候选；一次 best4 运行同时输出 candidate 0 的 single 视图和四选一的 best4 视图。汇总为 `result.json`，详细指标在 `views/single/scores/`、`views/best4/scores/`。若只研究单候选预算，使用 `--variant raw_single`。

## 原始历史续写版

64 故事／256 帧开发组：

```bash
python run.py metrics prepare --variant history_best4 --limit-stories 2 --gpus 0 --output outputs/history_smoke
python run.py metrics run --output outputs/history_smoke
```

去掉 `--limit-stories 2` 即使用完整开发组。每个故事独立维护选中的生成历史，下一帧只接收此前的图片和 caption。输出在 `frames/`，汇总在 `result.json`。

原 20-case 相邻片段／stride 8 实验：

```bash
python run.py pilot prepare --limit-cases 2 --gpus 0 --output outputs/pilot_smoke
python run.py pilot run --output outputs/pilot_smoke
```

去掉 `--limit-cases 2` 后保留原 20 cases、三条 5／5／4 帧分支、固定种子、共用首帧和分支独立历史：共 280 个逻辑帧、240 次独立帧生成、960 张候选。需要资源包中的 `pilot20/cases.json`。此入口生成 JSON 指标与全部候选，不重建旧实验中的其他方法对照海报。

## 接手阅读顺序

1. [版本与指标协议](docs/PROTOCOLS.md)：先确认要复现哪一行成绩。
2. [资源交接](docs/ASSETS.md)：准备模型、两套人物 checkpoint、mapper、图库和数据。
3. [训练与代码导读](docs/HANDOFF.md)：理解模块、原训练阶段、D1 组合和恢复条件。
4. [检查记录](docs/VALIDATION.md)：了解本次已验证范围及尚未重跑的部分。
5. [上传 GitHub](docs/GITHUB.md)：以本目录为仓库根目录上传。
6. [权重发布与下载](docs/WEIGHTS.md)：按版本安装权重，或维护 Releases。

所有运行只读取外部原始资源，在指定的新输出目录中写入结果。源码来源和包装层调整记录在 [SOURCE_MAP.json](SOURCE_MAP.json)。
