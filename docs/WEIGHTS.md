# 权重发布与下载

项目训练权重作为 GitHub Releases 附件发布，代码保留在 Git 仓库中。
发布仓库为 [liwk1616/qwen-klein-baseline](https://github.com/liwk1616/qwen-klein-baseline/releases)，
首个权重版本标签为 `weights-v1`。权重的角色、大小、许可证与限制见 [模型卡](MODEL_CARD.md)。

五个 safetensors 文件共 476,893,264 字节，约 454.80 MiB。单文件最大约
142.52 MiB，适合直接作为 Release 附件。GitHub Releases 允许每个附件小于
2 GiB，见 [官方说明](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)。

## 维护者：导出与发布

在原权重所在机器上，指定一个新的目录（不能在 Git 仓库或原始资产目录内部）：

```bash
python tools/weights.py export --source /path/to/QwenAnimeBaseline --output ../contract-baseline-weights-release
python tools/weights.py publish --directory ../contract-baseline-weights-release --check-only
```

这会复制五个原始 safetensors，生成安装清单、模型卡、许可证、来源声明和发布说明。
原始权重、配置和实验结果不改动。配置文本原样保存在清单中，安装时恢复。
清单明确此次仅提供推理文件；`complete.json` 中历史训练文件列表不表示那些文件也被发布。

先将新增的工具和文档提交、推送到 GitHub 的 `main`，再运行：

```bash
python tools/weights.py publish --directory ../contract-baseline-weights-release
```

脚本在终端隐藏输入 GitHub token；也支持已设置的 `GH_TOKEN` 或 `GITHUB_TOKEN`。
token 必须属于可写该仓库的账号，包含目标仓库的 **Contents: Read and write** 权限。
令牌不写入文件、不出现在命令参数或输出中。

发布流程先确认仓库公开且当前凭据有写权限，再创建草稿、逐个上传附件，确认附件名称、
长度和上传状态齐全后公开。失败会保留草稿；再次执行会续传尚未上传的附件。
已上传附件只按名称、长度和状态核对，不计算内容指纹；如果本地待上传内容有改动，
应使用新的发布目录和标签，而非复用旧草稿。冲突、未完整上传或意外附件会阻止公开，
已公开的同名版本不会被覆盖。

也可登录 GitHub 的 Releases 页面手工新建 `weights-v1`，将导出目录中除
`RELEASE_NOTES.md` 外的九个文件全部作为附件上传，将 `RELEASE_NOTES.md` 作为正文。
只有将附件上传并公开之后，下面的在线下载命令才可用。

## 使用者：安装

在源码仓库中执行：

```bash
python tools/weights.py install --variant all --output assets/QwenAnimeBaseline
```

只需要某一条 baseline 时使用 `--variant raw_best4` 或 `--variant history_best4`。
发布附件保持五个独立文件，安装器按清单放入原项目的相对目录布局，不要求手工改名。
所有下载、长度和张量布局检查完成后才安装；目标文件已存在时拒绝覆盖。

离线交接同一个导出目录时使用：

```bash
python tools/weights.py install --from-directory ../contract-baseline-weights-release --output assets/QwenAnimeBaseline
```

安装器只补齐本项目的推理 checkpoint。基础模型、数据、图库和评分器依照
[完整资源清单](ASSETS.md)准备。按 `configs/assets.example.json` 配置环境和路径后，
执行 `python run.py check --runtime`；单独安装权重后缺少其他资源的报错并不代表权重安装失败。

## 验证范围

导出、安装和上传使用显式文件清单、大小、可读 JSON 与张量名称、形状、类型、数据区长度检查，
不生成内容指纹。两条 baseline 的训练步数和 mapper 配对也会检查。
迁移验证、回归测试与真实权重安装检查记录在 [验证记录](VALIDATION.md)。
