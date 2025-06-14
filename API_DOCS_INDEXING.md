# Akasha 索引API详细文档

本文档详细描述了Akasha的索引相关API端点和状态报告。

## 索引流程

### 建立索引

```
POST /api/index
```

**响应**:

```json
{
  "success": true,
  "message": "索引过程已开始",
  "status": {
    "is_indexing": true,
    "progress": 10,
    "message": "准备索引...",
    "error": null,
    "stats": {
      "thumbnails": {},
      "indexing": {}
    }
  }
}
```

### 重建索引

```
POST /api/reindex
```

**响应**:

```json
{
  "success": true,
  "message": "重建索引过程已开始",
  "status": {
    "is_indexing": true,
    "progress": 10,
    "message": "重新初始化数据库...",
    "error": null,
    "stats": {
      "thumbnails": {},
      "indexing": {}
    }
  }
}
```

### 获取索引状态

```
GET /api/index/status
```

**响应示例 (进行中)**:

```json
{
  "is_indexing": true,
  "progress": 60,
  "message": "索引图像...",
  "error": null,
  "stats": {
    "thumbnails": {
      "total": 100,
      "success": 95,
      "failed": 5,
      "skipped": 0
    },
    "indexing": {}
  }
}
```

**响应示例 (完成，有失败)**:

```json
{
  "is_indexing": false,
  "progress": 100,
  "message": "索引完成，但有 3 张图片索引失败。新增索引 97 张，删除 0 张。",
  "warning": "共有 8 张图片处理失败，详见统计数据。",
  "error": null,
  "stats": {
    "thumbnails": {
      "total": 100,
      "success": 95,
      "failed": 5,
      "skipped": 0
    },
    "indexing": {
      "total_found": 100,
      "to_index": 100,
      "newly_indexed": 97,
      "failed_index": 3,
      "deleted": 0,
      "index_created": true
    }
  }
}
```

### 获取失败详情

```
GET /api/index/failures
```

**响应示例**:

```json
{
  "success": true,
  "total_failures": 8,
  "failures": [
    {
      "process": "thumbnail_creation",
      "file": "/workspaces/akasha/data/docs/image1.psd",
      "stage": "open_image",
      "error": "cannot identify image file"
    },
    {
      "process": "thumbnail_creation",
      "file": "/workspaces/akasha/data/docs/image2.jpg",
      "stage": "compress_and_save",
      "error": "disk space error"
    },
    {
      "process": "indexing",
      "file": "/workspaces/akasha/data/docs/image3.png",
      "stage": "embed_and_insert",
      "error": "API response timeout"
    }
  ]
}
```

## 错误处理策略

1. **缩略图创建过程中的错误**:
   - 单个图片处理失败不会中断整个缩略图创建流程
   - 失败的图片会被记录在`stats.thumbnails.failures`中
   - 总失败计数会在`stats.thumbnails.failed`中更新

2. **索引过程中的错误**:
   - 单个图片索引失败不会中断整个索引流程
   - 失败的图片会被记录在`stats.indexing.failures`中
   - 总失败计数会在`stats.indexing.failed_index`中更新

3. **整体索引流程的错误**:
   - 如果整个流程因异常中断，`error`字段会包含异常信息
   - 部分成功的操作仍会保存在数据库中
   - 任何个别图片失败的情况都不会设置`error`字段，但可能设置`warning`字段
