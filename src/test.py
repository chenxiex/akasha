def test_cohere_embed():
    import os
    from dotenv import load_dotenv
    import logging
    import cohere
    from embedder import cohere_embed
    import PIL.Image
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    CO_API_KEY = os.getenv("CO_API_KEY")
    if not CO_API_KEY:
        raise ValueError("COHERE_API_KEY not found in environment variables.")

    cohere_client = cohere.ClientV2(api_key=CO_API_KEY)
    embed_client = cohere_embed(cohere_client)
    embeddings = embed_client.embed_image(PIL.Image.open("data/docs/10.png"))
    print(embeddings[0:10])
    embedded_query = embed_client.embed_query("一张头像")
    print(embedded_query[0:10])


if __name__ == "__main__":
    pass
