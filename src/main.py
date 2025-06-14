import indexer
import queryer
import embedder
import db_init
import logging
import pathlib
import os

def main():
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    # 初始化数据库
    db_init.initialize_database()

    # 创建嵌入客户端
    import cohere
    CO_API_KEY = os.getenv("CO_API_KEY")
    if not CO_API_KEY:
        raise ValueError("CO_API_KEY not found in environment variables.")
    cohere_client = cohere.ClientV2(api_key=CO_API_KEY)
    embed_client = embedder.cohere_embed(cohere_client)

    # 创建索引器客户端
    indexer_client = indexer.indexer(embed_client)
    data_path = os.getenv("DATA_PATH", "data")
    data_path = pathlib.Path(data_path)
    if not data_path.exists():
        raise ValueError(f"Data path {data_path} does not exist.")
    
    # 创建缩略图并索引图片
    indexer_client.create_thumbnails(docs_path=data_path/"docs", thumbnails_path=data_path/"thumbnails")
    indexer_client.index_images(docs_path=data_path/"docs")

    # 创建查询器客户端
    queryer_client = queryer.queryer(embed_client)
    # 执行查询
    while True:
        query = input("输入查询内容 (或按回车退出): ")
        if not query:
            break
        results = queryer_client.query(query, max_dist=0.57)
        for result in results:
            print(f"图像: file://{data_path/'thumbnails'/result[0]} ，相似度: {result[1]:.4f}")

if __name__ == "__main__":
    main()