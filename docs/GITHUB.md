# 上传 GitHub

以 `contract-baseline-github` 本目录为仓库根目录。上传前运行：

```bash
python run.py check --source-only
python -B -m unittest discover -s tests -v
python tools/check_publish.py
```

`.gitignore` 已排除模型、数据、运行结果、环境目录、日志和 `configs/assets.local.json`。`results/` 只保存本仓库有意提供的小型历史指标摘要。`tools/check_publish.py` 检查拟提交的源码区是否混入大文件、本机挂载路径或常见凭证格式；它不是对所有类型凭证的保证。

在 GitHub 建立空仓库后，可在本目录执行：

```bash
git init
git symbolic-ref HEAD refs/heads/main
git add .
git status --short
git commit -m "Add frozen contract and history baselines"
git remote add origin git@github.com:YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

请把示例仓库地址替换成实际地址。以上初始化仅适用于新的本地仓库；已有 `origin` 时使用 `git remote set-url origin 仓库地址` 修改。当前交接仓库的地址为 `https://github.com/liwk1616/qwen-klein-baseline.git`。

项目训练权重作为 Releases 附件提供，不纳入普通 Git 提交，操作见 [权重发布与下载](WEIGHTS.md)。完整资源包中的基础模型、数据和第三方评价器按 [资源交接](ASSETS.md) 获取。

五个项目训练推理权重采用 Apache-2.0，范围见 [权重授权说明](../NOTICE-WEIGHTS.md)。项目代码与继承代码的授权单独记录；本次权重许可证不自动适用于整个代码库。第三方依赖、模型和数据保留各自来源和授权要求，见 [来源说明](../THIRD_PARTY.md)。
