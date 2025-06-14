import embedder
import logging
import db_init
from pgvector.psycopg2 import register_vector
import numpy as np

class queryer:
    embed:embedder.embed
    def __init__(self, embed:embedder.embed):
        self.embed = embed
    
    def query(self, query_text: str, max_dist:float) -> list[tuple]:
        '''
        查询与文本相关的图片

        :param query_text: 查询文本

        :return: 包含文件名和相似度的列表，格式为 [(file_name, similarity), ...]
        '''
        if not query_text:
            raise ValueError("Query text cannot be empty.")
        
        # 获取查询文本的嵌入向量
        query_vector = self.embed.embed_query(query_text)
        query_vector = np.array(query_vector)

        # 连接到数据库
        conn = db_init.get_db_connection()
        cur = conn.cursor()
        register_vector(cur)

        # 执行向量相似度查询，返回文件名和相似度
        cur.execute(
            "SELECT file_name, 1 - (embedding <=> %s) AS cosine_similarity FROM embeddings WHERE embedding <=> %s < %s ORDER BY cosine_similarity DESC",
            (query_vector, query_vector, max_dist)
        )

        results = cur.fetchall()
        if not results:
            logging.info("No results found for the query.")
            return []
        
        # 提取文件名和相似度
        result_pairs = [(row[0], row[1]) for row in results]
        logging.info(f"Found {len(result_pairs)} results for the query.")
        return result_pairs

def main():
    import cohere
    import os
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    CO_API_KEY = os.getenv("CO_API_KEY")
    if not CO_API_KEY:
        raise ValueError("CO_API_KEY not found in environment variables.")
    
    cohere_client = cohere.ClientV2(api_key=CO_API_KEY)
    embed_client = embedder.cohere_embed(cohere_client)
    queryer_client = queryer(embed_client)
    results = queryer_client.query("高楼大厦的图片", max_dist=1)
    print("查询结果:", results)

if __name__ == "__main__":
    main()