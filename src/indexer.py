import os
import PIL.Image
import embedder
import pathlib
import db_init
from pgvector.psycopg2 import register_vector
import logging
import numpy as np
import utilities

class indexer:
    embed:embedder.embed
    def __init__(self, embed:embedder.embed):
        '''
        :param embed: embedding client
        '''
        self.embed = embed
    
    def __create_index(self, cur):
        '''
        创建索引
        '''
        # 创建索引
        cur.execute(
            "CREATE INDEX IF NOT EXISTS embeddings_hnsw_idx ON embeddings USING hnsw (embedding vector_cosine_ops)"
        )

    def create_thumbnails(self, docs_path:pathlib.Path, thumbnails_path:pathlib.Path, recreate:bool=False):
        '''
        创建缩略图

        :param docs_path: 图片文件夹路径

        :param thumbnails_path: 缩略图文件夹路径
        '''
        thumbnail_size = 500 * 1024

        if not docs_path.exists():
            raise ValueError(f"Docs path {docs_path} does not exist.")
        # 确保缩略图目录存在
        thumbnails_path.mkdir(parents=True, exist_ok=True)
        # 生成缩略图
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(list(docs_path.glob(f'*{ext}')))
            image_files.extend(list(docs_path.glob(f'*{ext.upper()}')))
        for file in image_files:
            if not thumbnails_path.joinpath(file.name).exists() or recreate:
                with PIL.Image.open(file) as thumbnail:
                    try:
                        utilities.compress_image(thumbnail, thumbnail_size)
                        thumbnail_path = thumbnails_path / file.name
                        thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
                        thumbnail.save(thumbnail_path, format=thumbnail.format)
                    except Exception as e:
                        logging.error(f"Failed to create thumbnail for {file}: {e}")

    def index_images(self, docs_path:pathlib.Path):
        '''
        索引图片文件夹中的图片

        :param docs_path: 图片文件夹路径
        '''
        if not docs_path.exists():
            raise ValueError(f"Docs path {docs_path} does not exist.")

        conn=db_init.get_db_connection()
        cur = conn.cursor()
        register_vector(cur)

        # 获取目录下所有jpg和png文件
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(list(docs_path.glob(f'*{ext}')))
            image_files.extend(list(docs_path.glob(f'*{ext.upper()}')))

        # 查询已经在数据库中的文件名
        cur.execute("SELECT file_name FROM embeddings")
        indexed_file_names = [row[0] for row in cur.fetchall()]
        
        # 筛选出未索引的图片文件
        unindexed_files = [file for file in image_files if file.name not in indexed_file_names]

        # 筛选出已经删除的图片文件
        deleted_file_names = [file for file in indexed_file_names if not docs_path.joinpath(file).exists()]

        logging.info(f"{len(unindexed_files)} unindexed files found.")
        
        # 处理未索引的图片文件
        for file in unindexed_files:
            with PIL.Image.open(file) as image:
                if image.format not in ["JPEG", "PNG"]:
                    logging.warning(f"Unsupported image format: {image.format} for file {file}. Skipping.")
                    continue
                # 嵌入图片
                try:
                    logging.info(f"Indexing {file}...")
                    embedding_vector = self.embed.embed_image(image)
                    embedding_vector = np.array(embedding_vector)
                    cur.execute(
                        "INSERT INTO embeddings (file_name, embedding) VALUES (%s, %s)",
                        (file.name, embedding_vector)
                    )
                    conn.commit()
                except Exception as e:
                    logging.error(f"Failed to index {file}: {e}")
            
        # 删除数据库中已删除的文件记录
        if deleted_file_names:
            try:
                cur.execute(
                    "DELETE FROM embeddings WHERE file_name = ANY(%s)",
                    (deleted_file_names,)
                )
                conn.commit()
                logging.info(f"Deleted {len(deleted_file_names)} files from database that no longer exist.")
            except Exception as e:
                logging.error(f"Failed to delete files from database: {e}")
        
        # 创建索引
        try:
            self.__create_index(cur)
            conn.commit()
            logging.info("Index created successfully.")
        except Exception as e:  
            logging.error(f"Failed to create index: {e}")
            conn.rollback()
        
        cur.close()
        conn.close()

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