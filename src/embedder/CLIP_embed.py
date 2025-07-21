import clip
import torch
import torchvision
from typing import Callable, List
import PIL.Image
import logging
import numpy as np

class CLIP_embed:
    model_name: str = "ViT-B/32"
    model: clip.model.CLIP
    preprocess: Callable[[PIL.Image.Image], torch.Tensor]
    device: torch.device

    def __init__(self, model=None, preprocess=None, model_name: str = "ViT-B/32", device=None):
        '''
        :param model: CLIP model
        :param preprocess: CLIP preprocess
        :param model_name: a name of available CLIP models
        :param device: 设备类型，如果为None则自动选择（优先级：cuda > mps > cpu）
        '''
        self.model_name = model_name
        
        # 自动选择设备
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        if (model is None or preprocess is None):
            new_model, new_preprocess = clip.load(model_name, device=self.device)
            self.model=model if model is not None else new_model
            self.preprocess=preprocess if preprocess is not None else new_preprocess
        else:
            self.model = model
            self.preprocess = preprocess
        
        self.model.to(self.device).eval()

    def embed_image(self, image: PIL.Image.Image) -> List[float]:
        '''
        嵌入图片

        :param image: 图片对象

        :return: 嵌入向量

        :raises ValueError: 如果图片格式不受支持或无法识别
        '''
        try:
            image = image.convert("RGB")
            image_input = self.preprocess(image)
            image_input = torch.tensor(np.stack([image_input])).to(self.device)
        except Exception as e:
            logging.error(f"Embedding image failed: {e}.")
            raise ValueError("Image cannot be transformed into tensor")
        with torch.no_grad():
            image_features = self.model.encode_image(image_input).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features[0]

    def embed_query(self, text: str) -> List[float]:
        '''
        嵌入查询文本

        :param text: 查询文本

        :return: 嵌入向量
        '''
        text_input = clip.tokenize(text).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_input).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features[0]
