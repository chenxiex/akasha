import logging
import unittest
from typing import List

import PIL.Image
import torch

import embedder


class BaseEmbedTest(unittest.TestCase):
    """Base test class containing common embedding test methods."""

    def image_embed(self, embed_client: embedder.embed, image_path: str) -> List[float]:
        """Generic method to embed an image using any embed client."""
        embeddings = embed_client.embed_image(PIL.Image.open(image_path))
        return embeddings

    def query_embed(self, embed_client: embedder.embed, query: str) -> List[float]:
        """Generic method to embed a query using any embed client."""
        embeddings = embed_client.embed_query(query)
        return embeddings

    def generic_test_embed_image(self, embed_client: embedder.embed):
        image_path = "data/docs/1.jpg"
        embeddings = self.image_embed(
            embed_client, image_path)
        self.assertIsNotNone(embeddings)
        self.assertEqual(len(embeddings), 1024)

    def generic_test_embed_query(self, embed_client: embedder.embed):
        query = "A beautiful sunset over the mountains"
        embeddings = self.query_embed(embed_client, query)
        self.assertIsNotNone(embeddings)
        self.assertEqual(len(embeddings), 1024)


class TestCohereEmbed(BaseEmbedTest):
    embed_client: embedder.cohere_embed

    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        self.embed_client = embedder.cohere_embed()

    def test_embed_image(self):
        self.generic_test_embed_image(self.embed_client)

    def test_embed_query(self):
        self.generic_test_embed_query(self.embed_client)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
class TestCLIPEmbed(BaseEmbedTest):
    embed_client: embedder.CLIP_embed

    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        self.embed_client = embedder.CLIP_embed()

    def test_embed_image(self):
        self.generic_test_embed_image(self.embed_client)

    def test_embed_query(self):
        self.generic_test_embed_query(self.embed_client)


if __name__ == "__main__":
    unittest.main()
