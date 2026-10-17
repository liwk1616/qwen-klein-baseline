# 接手与训练说明

## 先完成一次小规模推理

先按 README 跑两个故事的 prepare，确认 `run_plan.json` 的 variant、split、checkpoint、候选数量和 GPU 分配，再运行 `run`。先检查每帧的实际 caption、参考、候选和选图记录，再看汇总数字。所有模型从本地资源加载，不需要在线 API 凭证。

| 代码 | 职责 |
| --- | --- |
| `run.py`、`runtime.py` | 环境选择、资源定位、子进程与路径迁移 |
| `run_metrics.py` | raw 完整 test 与 history 开发组的准备、调度、汇总 |
| `run_pilot.py`、`core/pilot_frame.py` | 原 pilot 的共用首帧、三分支和因果前缀 |
| `core/raw_generate.py` | D1 条件融合与逐帧候选生成 |
| `core/independent_candidate_protocol.py` | 原始描述、命名人物、种子与 raw 排序规则 |
| `core/raw_rank.py` | raw 候选的分类器与 Qwen 打分 |
| `core/models.py`、`current_bank.py` | 历史版模型加载、指代解析、mapper 与当帧图库条件 |
| `core/char_klein.py` | 人物融合层、人物训练监督 |
| `core/story_klein_mapper.py` | 原始历史条件 mapper |
| `core/stage_b_score.py` | 完整计划帧的分类器评价 |
| `core/train_aligned_klein.py` | 原人物与 mapper 训练流程 |
| `core/aligned_data.py`、`aligned_encoders.py` | 动态训练采样与在线编码 |
| `core/asset_paths.py` | 在内存中映射历史资产路径，保持权重文件不变 |

## 权重来源

合同指标版 D1 由两份历史 checkpoint 组成：C0 25k 的 `adapter_model.safetensors`，以及 B 26k 的 `character_fusion.safetensors`。B 从 C0 继续训练，训练时以 0.5 概率使用固定图库参考；D1 仅取 B 的融合层。它不是独立训练到“D1 step”的 checkpoint，也不是 fusion-only 训练所得。

历史版使用 C0 自身的 LoRA 与融合层，再加载与它配对训练的 8k mapper。不能把 raw 的 D1 直接放到该 mapper 前面。

本包训练核心取自 2026-09-20 的冻结快照，未带入后续 fusion-only、派生监督或 history-response 训练代码。训练参数见 `configs/training.json`。

## 训练入口

先准备训练补充资源。以下命令的 `--prepare-only` 仅打印实际启动参数；去掉它才会启动训练。

人物训练配置（有效 batch 32）：

```bash
python run.py train --stage character --gpus 0,1,2,3 \
  --output outputs/character_new --prepare-only
```

从完整 C0 25k 继续训练 B，沿原 27k 总预算先停在 26k：

```bash
python run.py train --stage reference --gpus 0,1,2,3 \
  --resume /path/to/full_C0_checkpoint_025000 \
  --output outputs/reference_B_new --prepare-only
```

初始学习率为 LoRA `5e-5`、融合层 `1e-4`；参考混合概率 0.5。此阶段恢复优化器、RNG、采样位置，并显式允许延长预算和改变参考策略。D1 推理目录不能代替完整 C0 恢复目录。

从配置中的 C0 25k 新训 mapper，沿原 20k 总预算停在 8k：

```bash
python run.py train --stage story --gpus 0,1,2,3 \
  --output outputs/mapper_new --prepare-only --stop-at 8000
```

需要使用新训练的人物 checkpoint 时，附 `--character-checkpoint /path/to/new_C0_checkpoint`。`--steps 8000` 与 `--steps 20000 --stop-at 8000` 的学习率进程不同，不应混写。

重训 C0/B 后组合新的 D1：

```bash
python tools/compose_d1.py --lora /path/to/C0_checkpoint_025000 \
  --fusion /path/to/B_checkpoint_026000 --output /path/to/new_D1
```

原历史训练曾迁移 GPU 布局，C0 记录保留逻辑 sampler 为 2×16，而后续物理执行为 4×8。这里给出原训练方法和配置入口；从头重训并不保证得到与历史权重逐张相同的生成结果。复现已记录指标时使用交接的冻结权重。

## 运行中断与输出

raw 的生成、classifier 排序、visual 排序、视图汇总是独立阶段；`metrics run` 顺序执行它们。`metrics rank-classifier`、`rank-visual`、`finish` 可用于执行尚未创建输出的阶段。已存在的阶段目录会被拒绝，不能靠重复执行 `run` 覆盖结果；最直接的重新运行方法是新建输出目录。

history 开发组同样要求新目录。pilot 按已经保存的 `selected.json` 恢复已完成帧，并核对 request 与前缀；它不保证复用中断时尚未完成的一帧候选。

准备后的 run_plan 和生成记录使用显式绝对路径。可在任意工作目录启动代码；已经开始的输出目录应留在原位置。本次迁移支持代码与外部输入资产换根目录，不自动搬迁进行中的运行。

## 后续研究

先保存这两条基线的配置和结果，再在单独实验目录开发新方法。比较时固定 split、故事清单、caption 协议、图库来源、候选预算、随机种子和评分方法。不要把开发集的 history 成绩与 raw 的 test 成绩拼成改进前后对照。
