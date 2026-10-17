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
git init -b main
git add .
git status --short
git commit -m "Add frozen contract and history baselines"
git remote add origin git@github.com:YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

请把示例仓库地址替换成实际地址。资源包单独交接，不纳入 Git 提交。本次整理只创建了本地目录，没有创建 GitHub 仓库、提交或推送。

项目代码的公开授权由项目负责人决定；本包未擅自添加 MIT、Apache 等项目许可证。第三方依赖、模型、数据和继承的上游代码保留各自来源和授权要求，见 [来源说明](../THIRD_PARTY.md)。
