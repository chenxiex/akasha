import time
import logging
from typing import List
import os

import cohere
import PIL.Image
from dotenv import load_dotenv

import utilities


class cohere_embed:
    client: cohere.ClientV2
    # 最大图片大小，单位为字节。部分API对图片大小限制为Tokens，即使很小的图片也可能超过限制，需要根据具体情况调整。
    image_max_size: int = 256*1024
    model_name: str = "Cohere-embed-v3-multilingual"
    retry_limit: int = 3
    retry_interval: int = 5

    def __init__(self, client: cohere.ClientV2 | None = None, model_name: str = "Cohere-embed-v3-multilingual"):
        '''
        :param client: cohere client
        '''
        if client is not None:
            self.client = client
        else:
            self.client = self.__create_cohere_client()
        self.model_name = model_name

    def __create_cohere_client(self, api_key: str = "") -> cohere.ClientV2:
        '''
        创建Cohere客户端

        :param api_key: Cohere API密钥

        :return: Cohere客户端实例

        :raises ValueError: 如果API密钥未提供且环境变量中未找到CO_API_KEY
        '''
        load_dotenv()

        if not api_key:
            CO_API_KEY = os.getenv("CO_API_KEY")
            if not CO_API_KEY:
                raise ValueError(
                    "CO_API_KEY not provided")
        else:
            CO_API_KEY = api_key
        cohere_client = cohere.ClientV2(api_key=CO_API_KEY)
        return cohere_client

    def embed_image(self, image: PIL.Image.Image) -> List[float]:
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
                logging.warning(
                    f"Embedding image failed: {e}. Retrying {i+1}/{self.retry_limit}...")
                time.sleep(self.retry_interval)
                continue
            break
        if response and response.embeddings and response.embeddings.float_:
            return response.embeddings.float_[0]
        else:
            raise ValueError("No embeddings returned from the API response.")

    def embed_query(self, text: str) -> List[float]:
        '''
        嵌入查询文本

        :param text: 查询文本

        :return: 嵌入向量
        '''
        response = self.client.embed(model=self.model_name,
                                     input_type="search_query",
                                     embedding_types=["float"],
                                     texts=[text])
        if response.embeddings and response.embeddings.float_:
            vec = response.embeddings.float_[0]
        else:
            raise ValueError("No embeddings returned from the API response.")
        return vec
