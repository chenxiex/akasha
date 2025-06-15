# Akasha 图片检索系统

Akasha图片检索系统是一个以文搜图的图片检索系统。

## 环境配置

建议使用 [Dev Containers](https://containers.dev/) 配置环境。

否则，使用 `conda` 参考 `.github/environment.yml` 创建 `python` 环境。然后，自行搭建 `PostgreSQL+pgvector` 数据库。

环境搭建完成后，参考 `.env.example` 配置环境变量。你可以使用 `.env` 配置环境变量。

## 使用

将需要检索的图片放入**数据目录/docs**中。数据目录可以通过环境变量 `DATA_PATH` 指定，也可以通过 Web UI 中的设置界面更改。例如，数据目录默认为 `data/`，此时应该将需要检索的图片放入 `data/docs/` 内，形成 `data/docs/1.jpg` 这样的结构。

要使用 Web UI：

```
python src/server.py
```

Web UI 支持在线更新索引。在初次运行时，需要点击“建立索引”建立索引，每次增删图片后都需要再次点击“建立索引”以获取最新结果。

要使用命令行界面（不建议，仅供调试用途）：

```
python src/main.py
```

命令行界面不支持在线更新索引。每次增删图片都需要重新启动以更新索引。