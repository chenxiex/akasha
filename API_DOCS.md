# Akasha API 文档

本文档描述了 Akasha 图像搜索系统提供的API端点。

## 健康检查

### 检查API服务健康状态

```
GET /api/health
```

**响应**:

```json
{
  "status": "ok"
}
```

## 索引管理

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
    "error": null
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
    "error": null
  }
}
```

### 获取索引状态

```
GET /api/index/status
```

**响应**:

```json
{
  "is_indexing": true,
  "progress": 60,
  "message": "索引图像...",
  "error": null
}
```

## 设置管理

### 获取所有设置项

```
GET /api/settings
```

**响应**:

```json
{
  "success": true,
  "settings": {
    "data_path": "/workspaces/akasha/data"
  }
}
```

### 获取单个设置项

```
GET /api/settings/data_path
```

**响应 (存在)**:

```json
{
  "success": true,
  "key": "data_path",
  "value": "/workspaces/akasha/data"
}
```

**响应 (使用默认值)**:

```json
{
  "success": true,
  "key": "data_path",
  "value": "/workspaces/akasha/data",
  "default": true
}
```

**响应 (不存在)**:

```json
{
  "success": false,
  "message": "设置项 'unknown_key' 不存在",
  "key": "unknown_key"
}
```

### 更新设置项

```
POST /api/settings
```

**请求体**:

```json
{
  "data_path": "/new/path/to/data",
  "another_setting": "value"
}
```

**响应 (全部成功)**:

```json
{
  "success": true,
  "updated": {
    "data_path": "/new/path/to/data",
    "another_setting": "value"
  }
}
```

**响应 (部分失败)**:

```json
{
  "success": false,
  "updated": {
    "another_setting": "value"
  },
  "failed": {
    "data_path": "更新失败"
  }
}
```

## 图像搜索

### 搜索图像

```
GET /api/search?q=查询文本&max_dist=0.57
```

**参数**:

- `q`: 搜索查询文本
- `max_dist`: (可选) 最大距离阈值，默认为0.57

**响应**:

```json
{
  "success": true,
  "query": "查询文本",
  "results": [
    {
      "file_name": "image1.jpg",
      "similarity": 0.85
    },
    {
      "file_name": "image2.png",
      "similarity": 0.75
    }
  ]
}
```

### 获取图像缩略图

```
GET /api/thumbnail/image1.jpg
```

**响应**: 图像文件（JPEG或PNG格式）

### 获取Base64编码的缩略图

```
GET /api/thumbnail_base64/image1.jpg
```

**响应**:

```json
{
  "success": true,
  "file_name": "image1.jpg",
  "thumbnail": "data:image/jpeg;base64,/9j/4AAQSkZ..."
}
```

## 错误响应格式

所有API错误都会返回统一的JSON格式：

```json
{
  "success": false,
  "message": "错误描述"
}
```

HTTP状态码将根据错误类型适当设置。
