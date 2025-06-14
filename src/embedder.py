import PIL.Image
import utilities
import cohere
from typing import Protocol
import time
import logging

class embed(Protocol):
    def embed_image(self, image: PIL.Image.Image) -> list[float]:
        '''
        嵌入图片

        :param image: 图片对象

        :return: 嵌入向量
        '''
        return list[float]()

    def embed_query(self, text: str) -> list[float]:
        '''
        嵌入查询文本

        :param text: 查询文本

        :return: 嵌入向量
        '''
        return list[float]()

class cohere_embed:
    client:cohere.ClientV2
    image_max_size:int=1*1024*1024
    model_name:str
    # 每一个请求携带的图片数
    retry_limit:int=3
    retry_interval:int=5

    def __init__(self, client:cohere.ClientV2, model_name:str="Cohere-embed-v3-multilingual"):
        '''
        :param client: cohere client
        '''
        self.client = client
        self.model_name = model_name
    
    def embed_image(self, image:PIL.Image.Image):
        '''
        嵌入图片

        :param image: 图片对象

        :return: 嵌入向量

        :raises ValueError: 如果图片格式不受支持或无法识别
        '''
        thumbnail = [utilities.thumbnail_url(image, self.image_max_size)]
        response = None
        for i in range(self.retry_limit):
            try:
                response = self.client.embed(model=self.model_name,
                                            input_type="image",
                                            embedding_types=["float"],
                                            images=thumbnail)
            except Exception as e:
                logging.warning(f"Embedding image failed: {e}. Retrying {i+1}/{self.retry_limit}...")
                time.sleep(self.retry_interval)
                continue
            break
        if response and response.embeddings and response.embeddings.float_:
            return response.embeddings.float_[0]
        else:
            raise ValueError("No embeddings returned from the API response.")

    def embed_query(self, text: str, model_name:str="Cohere-embed-v3-multilingual") -> list[float]:
        '''
        嵌入查询文本

        :param text: 查询文本

        :param model_name: 模型名称

        :return: 嵌入向量
        '''
        response = self.client.embed(model=model_name,
                                     input_type="search_query",
                                     embedding_types=["float"],
                                     texts=[text])
        if response.embeddings and response.embeddings.float_:
            vec = response.embeddings.float_[0]
        else:
            raise ValueError("No embeddings returned from the API response.")
        return vec

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
 
    CO_API_KEY = os.getenv("CO_API_KEY")
    if not CO_API_KEY:
        raise ValueError("COHERE_API_KEY not found in environment variables.")

    cohere_client = cohere.ClientV2(api_key=CO_API_KEY)
    embed_client = cohere_embed(cohere_client)
    embeddings = embed_client.embed_image(PIL.Image.open("data/docs/10.png"))
    print(embeddings[0:10])  # 打印前10个元素
    embedded_query = embed_client.embed_query("一张头像")
    print(embedded_query[0:10])