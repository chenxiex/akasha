import PIL.Image
from typing import Protocol


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
