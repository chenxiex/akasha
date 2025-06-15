import os
import PIL.Image
import embedder
import pathlib
import db_init
from pgvector.psycopg2 import register_vector
import logging
import numpy as np
import utilities
from typing import Dict, List, Tuple, Any

class indexer:
    embed:embedder.embed
    def __init__(self, embed:embedder.embed) -> None:
        '''
        :param embed: embedding client
        '''
        self.embed = embed
    
    def __create_index(self, cur) -> None:
        '''
        创建索引
        '''
        cur.execute(
            "CREATE INDEX IF NOT EXISTS embeddings_hnsw_idx ON embeddings USING hnsw (embedding vector_cosine_ops)"
        )

    def create_thumbnails(self, docs_path:pathlib.Path, thumbnails_path:pathlib.Path, recreate:bool=False) -> Dict[str, Any]:
        '''
        创建缩略图

        :param docs_path: 图片文件夹路径
        :param thumbnails_path: 缩略图文件夹路径
        :param recreate: 是否重新创建已存在的缩略图

        :return: 包含处理结果统计的字典，格式为 {'total': int, 'success': int, 'deleted': int, 'failed': int, 'skipped': int, 'failures': list}
        '''
        thumbnail_size = 500 * 1024
        result = {
            'total': 0,
            'success': 0,
            'deleted': 0,
            'failed': 0,
            'skipped': 0,
            'failures': []
        }

        if not docs_path.exists():
            raise ValueError(f"Docs path {docs_path} does not exist.")
        
        thumbnails_path.mkdir(parents=True, exist_ok=True)
        
        # 生成缩略图
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(list(docs_path.glob(f'*{ext}')))
            image_files.extend(list(docs_path.glob(f'*{ext.upper()}')))
        
        result['total'] = len(image_files)
        
        for i, file in enumerate(image_files):
            thumbnail_path = thumbnails_path / file.name
            
            if thumbnail_path.exists() and not recreate:
                result['skipped'] += 1
                continue
                
            try:
                with PIL.Image.open(file) as thumbnail:
                    try:
                        utilities.compress_image(thumbnail, thumbnail_size)
                        thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
                        thumbnail.save(thumbnail_path, format=thumbnail.format)
                        logging.info(f"Created thumbnail for {file.name} ({i+1}/{result['total']})")
                        result['success'] += 1
                    except Exception as e:
                        error_msg = str(e)
                        logging.error(f"Failed to create thumbnail for {file}: {error_msg}")
                        result['failed'] += 1
                        result['failures'].append({
                            'file': str(file),
                            'stage': 'compress_and_save',
                            'error': error_msg
                        })
            except Exception as e:
                error_msg = str(e)
                logging.error(f"Failed to open image file {file}: {error_msg}")
                result['failed'] += 1
                result['failures'].append({
                    'file': str(file),
                    'stage': 'open_image',
                    'error': error_msg
                })
        
        # 移除已删除的缩略图
        existing_thumbnails = set(thumbnails_path.glob('*'))
        for existing_thumbnail in existing_thumbnails:
            if not existing_thumbnail.name in [file.name for file in image_files]:
                try:
                    existing_thumbnail.unlink()
                    logging.info(f"Deleted thumbnail {existing_thumbnail.name} as the original image no longer exists.")
                    result['deleted'] = result.get('deleted', 0) + 1
                except Exception as e:
                    error_msg = str(e)
                    logging.error(f"Failed to delete thumbnail {existing_thumbnail.name}: {error_msg}")
                    result['failures'].append({
                        'file': str(existing_thumbnail),
                        'stage': 'delete_thumbnail',
                        'error': error_msg
                    })
        return result

    def index_images(self, docs_path:pathlib.Path, recreate:bool=False) -> Dict[str, Any]:
        '''
        索引图片文件夹中的图片

        :param docs_path: 图片文件夹路径

        :return: 包含处理结果统计的字典，格式为 
                {'total_found': int, 'newly_indexed': int, 'failed_index': int, 
                 'deleted': int, 'index_created': bool, 'failures': list}
        '''
        result = {
            'total_found': 0,
            'newly_indexed': 0,
            'failed_index': 0,
            'deleted': 0,
            'index_created': False,
            'failures': []
        }

        if not docs_path.exists():
            raise ValueError(f"Docs path {docs_path} does not exist.")

        conn = db_init.get_db_connection()
        cur = conn.cursor()
        register_vector(cur)

        if recreate:
            try:
                # 删除现有表
                cur.execute("DROP TABLE IF EXISTS embeddings;")
                conn.commit()
                db_init.initialize_database()
            except Exception as e:
                logging.error(f"Failed to recreate embeddings table: {e}")
                conn.rollback()

        # 获取目录下所有jpg和png文件
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(list(docs_path.glob(f'*{ext}')))
            image_files.extend(list(docs_path.glob(f'*{ext.upper()}')))

        result['total_found'] = len(image_files)

        # 查询已经在数据库中的文件名
        cur.execute("SELECT file_name FROM embeddings")
        indexed_file_names = [row[0] for row in cur.fetchall()]
        
        # 筛选出未索引的图片文件
        unindexed_files = [file for file in image_files if file.name not in indexed_file_names]

        # 筛选出已经删除的图片文件
        deleted_file_names = [file for file in indexed_file_names if not docs_path.joinpath(file).exists()]

        logging.info(f"{len(unindexed_files)} unindexed files found.")
        
        # 处理未索引的图片文件
        for i, file in enumerate(unindexed_files):
            try:
                with PIL.Image.open(file) as image:
                    if image.format not in ["JPEG", "PNG"]:
                        logging.warning(f"Unsupported image format: {image.format} for file {file}. Skipping.")
                        result['failures'].append({
                            'file': str(file),
                            'stage': 'check_format',
                            'error': f"Unsupported image format: {image.format}"
                        })
                        result['failed_index'] += 1
                        continue
                    
                    # 嵌入图片
                    try:
                        logging.info(f"Indexing {file.name} ({i+1}/{len(unindexed_files)})...")
                        embedding_vector = self.embed.embed_image(image)
                        embedding_vector = np.array(embedding_vector)
                        cur.execute(
                            "INSERT INTO embeddings (file_name, embedding) VALUES (%s, %s)",
                            (file.name, embedding_vector)
                        )
                        conn.commit()
                        result['newly_indexed'] += 1
                    except Exception as e:
                        error_msg = str(e)
                        logging.error(f"Failed to index {file}: {error_msg}")
                        result['failures'].append({
                            'file': str(file),
                            'stage': 'embed_and_insert',
                            'error': error_msg
                        })
                        result['failed_index'] += 1
                        conn.rollback()
            except Exception as e:
                error_msg = str(e)
                logging.error(f"Failed to index image file {file}: {error_msg}")
                result['failures'].append({
                    'file': str(file),
                    'stage': 'open_file',
                    'error': error_msg
                })
                result['failed_index'] += 1
            
        # 删除数据库中已删除的文件记录
        if deleted_file_names:
            try:
                cur.execute(
                    "DELETE FROM embeddings WHERE file_name = ANY(%s)",
                    (deleted_file_names,)
                )
                conn.commit()
                logging.info(f"Deleted {len(deleted_file_names)} files from database that no longer exist.")
                result['deleted'] = len(deleted_file_names)
            except Exception as e:
                error_msg = str(e)
                logging.error(f"Failed to delete files from database: {error_msg}")
                result['failures'].append({
                    'stage': 'delete_records',
                    'error': error_msg
                })
                conn.rollback()
        
        # 创建索引
        try:
            self.__create_index(cur)
            conn.commit()
            logging.info("Index created successfully.")
            result['index_created'] = True
        except Exception as e:  
            error_msg = str(e)
            logging.error(f"Failed to create index: {error_msg}")
            result['failures'].append({
                'stage': 'create_index',
                'error': error_msg
            })
            conn.rollback()
        
        cur.close()
        conn.close()
        
        return result

def main():
    import cohere
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    CO_API_KEY = os.getenv("CO_API_KEY")
    if not CO_API_KEY:
        raise ValueError("CO_API_KEY not found in environment variables.")
    
    data_path = os.getenv("DATA_PATH", "data")
    data_path = pathlib.Path(data_path)
    if not data_path.exists():
        raise ValueError(f"Data path {data_path} does not exist.")

    cohere_client = cohere.ClientV2(api_key=CO_API_KEY)
    embed_client = embedder.cohere_embed(cohere_client)
    db_init.initialize_database()
    indexer_client = indexer(embed_client)
    indexer_client.create_thumbnails(docs_path=data_path/"docs", thumbnails_path=data_path/"thumbnails")
    indexer_client.index_images(docs_path=data_path/"docs")

if __name__ == "__main__":
    main()