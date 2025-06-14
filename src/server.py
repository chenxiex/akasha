from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import pathlib
import base64
import logging
import indexer
import queryer
import embedder
import db_init
import cohere
from dotenv import load_dotenv
import PIL.Image
import io
import threading
import time

# 加载环境变量
load_dotenv()
logging.basicConfig(level=logging.INFO)

# 初始化 Flask 应用
app = Flask(__name__, static_folder=os.path.join(os.getcwd(), "static"))
CORS(app)  # 启用CORS，允许前端跨域访问

# 全局变量用于追踪索引构建状态
indexing_status = {
    "is_indexing": False,
    "progress": 0,
    "message": "",
    "error": None,
    "stats": {
        "thumbnails": {},
        "indexing": {}
    }
}

# 全局变量用于存储客户端和路径
cohere_client:cohere.ClientV2
embed_client:embedder.embed
indexer_client:indexer.indexer
queryer_client:queryer.queryer
data_path:pathlib.Path
docs_path:pathlib.Path
thumbnails_path:pathlib.Path

def get_db_setting(key, default_value=None):
    """从数据库获取设置项"""
    try:
        conn = db_init.get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT value FROM settings WHERE key = %s", (key,))
        result = cur.fetchone()
        cur.close()
        conn.close()
        
        if result:
            return result[0]
        return default_value
    except Exception as e:
        logging.error(f"获取设置失败: {e}")
        return default_value

def update_db_setting(key, value):
    """更新数据库设置项"""
    try:
        conn = db_init.get_db_connection()
        cur = conn.cursor()
        
        # 使用UPSERT语法 (INSERT ... ON CONFLICT DO UPDATE)
        cur.execute(
            """
            INSERT INTO settings (key, value, updated_at) 
            VALUES (%s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (key) 
            DO UPDATE SET value = EXCLUDED.value, updated_at = CURRENT_TIMESTAMP
            """, 
            (key, value)
        )
        
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        logging.error(f"更新设置失败: {e}")
        return False

def initialize_server(hot_reload=False):
    """初始化服务器，包括数据库、客户端和目录"""
    global cohere_client, embed_client, indexer_client, queryer_client
    global data_path, docs_path, thumbnails_path
    
    try:
        if not hot_reload:
            logging.info("正在初始化数据库...")
            db_init.initialize_database(recreate=False)
        
        # 数据路径设置
        db_data_path = get_db_setting("data_path")
        env_data_path = os.getenv("DATA_PATH", "data")
        
        data_path_str = db_data_path if db_data_path else env_data_path
        data_path = pathlib.Path(data_path_str)
        
        if not data_path.is_absolute():
            data_path = pathlib.Path(os.getcwd()) / data_path

        docs_path = data_path / "docs"
        thumbnails_path = data_path / "thumbnails"
        
        docs_path.mkdir(parents=True, exist_ok=True)
        thumbnails_path.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"数据目录设置为: {data_path}")
        
        # 如果数据库中没有数据路径设置，则保存当前的数据路径
        if not db_data_path:
            update_db_setting("data_path", str(data_path))

        # 初始化各个对象
        if not hot_reload:
            CO_API_KEY = os.getenv("CO_API_KEY")
            if not CO_API_KEY:
                raise ValueError("CO_API_KEY not found in environment variables.")
            cohere_client = cohere.ClientV2(api_key=CO_API_KEY)

        embed_client = embedder.cohere_embed(cohere_client)
        indexer_client = indexer.indexer(embed_client)
        queryer_client = queryer.queryer(embed_client)
        
        logging.info("服务器初始化完成")
        return True
    except Exception as e:
        logging.error(f"服务器初始化失败: {e}")
        return False

def build_index_process(recreate=False):
    """索引创建过程，在单独的线程中运行"""
    global indexing_status
    
    try:
        indexing_status["is_indexing"] = True
        indexing_status["progress"] = 10
        
        if recreate:
            indexing_status["message"] = "重新初始化数据库..."
            # 重建索引时需要重新初始化数据库
            db_init.initialize_database(recreate=True)
        else:
            indexing_status["message"] = "准备索引..."
        
        indexing_status["progress"] = 30
        indexing_status["message"] = "创建缩略图..."
        
        # 创建缩略图，并获取统计结果
        thumbnail_stats = indexer_client.create_thumbnails(
            docs_path=docs_path, 
            thumbnails_path=thumbnails_path,
            recreate=recreate
        )
        
        indexing_status["stats"]["thumbnails"] = thumbnail_stats
        total_thumbnails = thumbnail_stats.get('total', 0)
        failed_thumbnails = thumbnail_stats.get('failed', 0)
        
        # 更新进度消息，包含成功/失败信息
        if failed_thumbnails > 0:
            indexing_status["message"] = f"缩略图创建完成，但有 {failed_thumbnails}/{total_thumbnails} 张图片处理失败。"
        else:
            indexing_status["message"] = f"缩略图创建完成，共处理 {total_thumbnails} 张图片。"
        
        indexing_status["progress"] = 60
        indexing_status["message"] = "索引图像..."
        
        # 索引图像，并获取统计结果
        indexing_stats = indexer_client.index_images(docs_path=docs_path)
        indexing_status["stats"]["indexing"] = indexing_stats
        
        # 计算总结果
        newly_indexed = indexing_stats.get('newly_indexed', 0)
        failed_index = indexing_stats.get('failed_index', 0)
        
        indexing_status["progress"] = 100
        
        # 添加详细的完成信息
        if failed_index > 0:
            indexing_status["message"] = f"索引完成，但有 {failed_index} 张图片索引失败。新增索引 {newly_indexed} 张，删除 {indexing_stats.get('deleted', 0)} 张。"
        else:
            indexing_status["message"] = f"索引完成。新增索引 {newly_indexed} 张，删除 {indexing_stats.get('deleted', 0)} 张。"
            
        # 如果有任何失败，设置一个警告标记
        if failed_thumbnails > 0 or failed_index > 0:
            indexing_status["warning"] = f"共有 {failed_thumbnails + failed_index} 张图片处理失败，详见统计数据。"
        
    except Exception as e:
        indexing_status["error"] = str(e)
        indexing_status["message"] = f"索引失败: {str(e)}"
        logging.error(f"索引过程出错: {e}")
    finally:
        indexing_status["is_indexing"] = False

# 静态文件路由
@app.route('/')
def index():
    static_folder = app.static_folder or os.path.join(os.getcwd(), "static")
    return send_from_directory(static_folder, 'index.html')

@app.route('/<path:path>')
def static_files(path):
    static_folder = app.static_folder or os.path.join(os.getcwd(), "static")
    return send_from_directory(static_folder, path)

@app.route('/api/index', methods=['POST'])
def build_index():
    """建立索引的API端点"""
    global indexing_status
    
    if indexing_status["is_indexing"]:
        return jsonify({
            "success": False,
            "message": "已经有一个索引过程在进行中",
            "status": indexing_status
        }), 400
    
    # 启动索引过程（在后台线程中）
    thread = threading.Thread(target=build_index_process, args=(False,))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        "success": True,
        "message": "索引过程已开始",
        "status": indexing_status
    })

@app.route('/api/reindex', methods=['POST'])
def rebuild_index():
    """重建索引的API端点"""
    global indexing_status
    
    if indexing_status["is_indexing"]:
        return jsonify({
            "success": False,
            "message": "已经有一个索引过程在进行中",
            "status": indexing_status
        }), 400
    
    # 启动重建索引过程（在后台线程中）
    thread = threading.Thread(target=build_index_process, args=(True,))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        "success": True,
        "message": "重建索引过程已开始",
        "status": indexing_status
    })

@app.route('/api/index/status', methods=['GET'])
def get_index_status():
    """获取索引状态的API端点"""
    global indexing_status
    return jsonify(indexing_status)

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """获取所有设置的API端点"""
    try:
        conn = db_init.get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT key, value FROM settings")
        
        settings = {}
        for row in cur.fetchall():
            settings[row[0]] = row[1]
        
        cur.close()
        conn.close()
        
        # 处理默认值
        if "data_path" not in settings:
            settings["data_path"] = str(data_path)
        
        return jsonify({
            "success": True, 
            "settings": settings
        })
    except Exception as e:
        logging.error(f"获取设置失败: {e}")
        return jsonify({
            "success": False,
            "message": f"获取设置失败: {str(e)}"
        }), 500

@app.route('/api/settings/<string:key>', methods=['GET'])
def get_setting(key):
    """获取单个设置项的API端点"""
    value = get_db_setting(key)
    
    if value is not None:
        return jsonify({
            "success": True,
            "key": key,
            "value": value
        })
    else:
        if key == "data_path":
            return jsonify({
                "success": True,
                "key": key,
                "value": str(data_path),
                "default": True
            })
        return jsonify({
            "success": False,
            "message": f"设置项 '{key}' 不存在",
            "key": key
        }), 404

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """更新设置的API端点"""
    try:
        data = request.get_json()
        
        if not data or not isinstance(data, dict):
            return jsonify({
                "success": False,
                "message": "无效的请求数据"
            }), 400
        
        updated = {}
        failed = {}
        
        for key, value in data.items():
            if update_db_setting(key, value):
                updated[key] = value
            else:
                failed[key] = "更新失败"
        
        result = {
            "success": len(failed) == 0,
            "updated": updated
        }
        
        if failed:
            result["failed"] = failed
        elif not initialize_server(hot_reload=True):
            result["message"] = "设置更新成功，但服务器重新加载失败，请检查日志"
            return jsonify(result), 500
            
        return jsonify(result)
    
    except Exception as e:
        logging.error(f"更新设置失败: {e}")
        return jsonify({
            "success": False,
            "message": f"更新设置失败: {str(e)}"
        }), 500

@app.route('/api/search', methods=['GET'])
def search_images():
    """搜索图像的API端点"""
    query = request.args.get('q', '')
    max_dist = float(request.args.get('max_dist', 0.57))
    
    if not query:
        return jsonify({
            "success": False,
            "message": "查询不能为空"
        }), 400
    
    try:
        # 执行查询
        results = queryer_client.query(query, max_dist=max_dist)
        response_results = []
        
        for file_name, similarity in results:
            thumbnail_path = thumbnails_path / file_name
            if thumbnail_path.exists():
                response_results.append({
                    "file_name": file_name,
                    "similarity": similarity,
                })
        
        return jsonify({
            "success": True,
            "query": query,
            "results": response_results
        })
        
    except Exception as e:
        logging.error(f"搜索出错: {e}")
        return jsonify({
            "success": False,
            "message": f"搜索出错: {str(e)}"
        }), 500

@app.route('/api/thumbnail/<path:file_name>', methods=['GET'])
def get_thumbnail(file_name):
    """获取缩略图的API端点"""
    try:
        # 安全地解析文件路径，防止目录遍历攻击
        file_path = (thumbnails_path / file_name).resolve()
        
        # 确保文件确实位于thumbnails目录下
        try:
            file_path.relative_to(thumbnails_path)
        except ValueError:
            # 如果文件不在thumbnails目录下，则是目录遍历尝试
            raise ValueError("路径遍历检测: 文件不在thumbnails目录中")
        
        if not file_path.exists():
            return jsonify({
                "success": False,
                "message": "找不到缩略图文件"
            }), 404
        
        # 确定文件的MIME类型
        mime_type = 'image/jpeg' if file_name.lower().endswith(('.jpg', '.jpeg')) else 'image/png'
        
        # 发送文件，而不是base64编码
        return send_file(str(file_path), mimetype=mime_type)
    
    except ValueError as e:
        logging.warning(f"安全警告 - 可能的路径遍历尝试: {e}")
        return jsonify({
            "success": False,
            "message": "无效的文件路径请求"
        }), 403
    except Exception as e:
        logging.error(f"获取缩略图出错: {e}")
        return jsonify({
            "success": False,
            "message": f"获取缩略图出错: {str(e)}"
        }), 500

@app.route('/api/thumbnail_base64/<path:file_name>', methods=['GET'])
def get_thumbnail_base64(file_name):
    """获取base64编码缩略图的API端点"""
    try:
        # 安全地解析文件路径，防止目录遍历攻击
        file_path = (thumbnails_path / file_name).resolve()
        
        # 确保文件确实位于thumbnails目录下
        try:
            file_path.relative_to(thumbnails_path)
        except ValueError:
            # 如果文件不在thumbnails目录下，则是目录遍历尝试
            raise ValueError("路径遍历检测: 文件不在thumbnails目录中")
        
        if not file_path.exists():
            return jsonify({
                "success": False, 
                "message": "找不到缩略图文件"
            }), 404
        
        # 读取文件并转换为base64
        with open(file_path, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
            img_format = 'jpeg' if file_name.lower().endswith(('.jpg', '.jpeg')) else 'png'
            img_src = f"data:image/{img_format};base64,{img_data}"
            
            return jsonify({
                "success": True,
                "file_name": file_name,
                "thumbnail": img_src
            })
    
    except ValueError as e:
        logging.warning(f"安全警告 - 可能的路径遍历尝试: {e}")
        return jsonify({
            "success": False,
            "message": "无效的文件路径请求"
        }), 403
    except Exception as e:
        logging.error(f"获取缩略图base64出错: {e}")
        return jsonify({
            "success": False,
            "message": f"获取缩略图base64出错: {str(e)}"
        }), 500

@app.route('/api/original/<path:file_name>', methods=['GET'])
def get_original_image(file_name):
    """获取原始图像的API端点"""
    try:
        # 安全地解析文件路径，防止目录遍历攻击
        file_path = (docs_path / file_name).resolve()
        
        # 确保文件确实位于docs目录下
        try:
            file_path.relative_to(docs_path)
        except ValueError:
            # 如果文件不在docs目录下，则是目录遍历尝试
            raise ValueError("路径遍历检测: 文件不在docs目录中")
        
        if not file_path.exists():
            return jsonify({
                "success": False,
                "message": "找不到原始图像文件"
            }), 404
        
        # 确定文件的MIME类型
        mime_type = 'image/jpeg' if file_name.lower().endswith(('.jpg', '.jpeg')) else 'image/png'
        
        # 设置响应头，让文件以附件形式下载
        return send_file(str(file_path), mimetype=mime_type, 
                        as_attachment=request.args.get('download', 'false').lower() == 'true',
                        download_name=file_name if request.args.get('download', 'false').lower() == 'true' else None)
    
    except ValueError as e:
        logging.warning(f"安全警告 - 可能的路径遍历尝试: {e}")
        return jsonify({
            "success": False,
            "message": "无效的文件路径请求"
        }), 403
    except Exception as e:
        logging.error(f"获取原始图像出错: {e}")
        return jsonify({
            "success": False,
            "message": f"获取原始图像出错: {str(e)}"
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({"status": "ok"})

@app.errorhandler(404)
def not_found(error):
    """处理404错误"""
    if request.path.startswith('/api/'):
        return jsonify({
            "success": False,
            "message": "API端点不存在"
        }), 404
    static_folder = app.static_folder or os.path.join(os.getcwd(), "static")
    return send_from_directory(static_folder, "index.html")

@app.errorhandler(500)
def internal_server_error(error):
    """处理500错误"""
    logging.error(f"服务器错误: {error}")
    return jsonify({
        "success": False,
        "message": "服务器内部错误"
    }), 500

@app.route('/api/index/failures', methods=['GET'])
def get_index_failures():
    """获取索引失败的详细信息"""
    global indexing_status
    
    failures = []
    
    # 收集缩略图创建失败的条目
    if 'thumbnails' in indexing_status["stats"] and 'failures' in indexing_status["stats"]["thumbnails"]:
        for failure in indexing_status["stats"]["thumbnails"]["failures"]:
            failures.append({
                "process": "thumbnail_creation",
                **failure
            })
    
    # 收集索引失败的条目
    if 'indexing' in indexing_status["stats"] and 'failures' in indexing_status["stats"]["indexing"]:
        for failure in indexing_status["stats"]["indexing"]["failures"]:
            failures.append({
                "process": "indexing",
                **failure
            })
    
    return jsonify({
        "success": True,
        "total_failures": len(failures),
        "failures": failures
    })

# 初始化服务器
if not initialize_server():
    logging.error("服务器初始化失败，退出程序")
    exit(1)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(host='0.0.0.0', port=port, debug=debug)
