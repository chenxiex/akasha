import PIL.Image
from typing import Protocol, List


class embed(Protocol):
    def embed_image(self, image: PIL.Image.Image) -> List[float]:
        '''
        嵌入图片

        :param image: 图片对象

        :return: 嵌入向量
        '''
        ...

    def embed_query(self, text: str) -> List[float]:
        '''
        嵌入查询文本

        :param text: 查询文本

        :return: 嵌入向量
        '''
        ...
