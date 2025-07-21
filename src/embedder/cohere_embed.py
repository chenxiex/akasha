import cohere
import PIL.Image
import utilities
import time
import logging
from typing import List


class cohere_embed:
    client: cohere.ClientV2
    # 最大图片大小，单位为字节。部分API对图片大小限制为Tokens，即使很小的图片也可能超过限制，需要根据具体情况调整。
    image_max_size: int = 256*1024
    model_name: str = "Cohere-embed-v3-multilingual"
    retry_limit: int = 3
    retry_interval: int = 5

    def __init__(self, client: cohere.ClientV2, model_name: str = "Cohere-embed-v3-multilingual"):
        '''
        :param client: cohere client
        '''
        self.client = client
        self.model_name = model_name

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
