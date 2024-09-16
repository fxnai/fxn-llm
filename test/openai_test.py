# 
#   Function
#   Copyright © 2024 NatML Inc. All Rights Reserved.
#

from fxn_llm import locally
from openai import OpenAI
from openai.types import CreateEmbeddingResponse

def test_create_embeddings_locally ():
    openai = OpenAI()
    openai = locally(openai)
    tag = "@nomic/nomic-embed-text-v1.5-quant"
    embedding = openai.embeddings.create(
        model=tag,
        input="search_query: What is the capital of France?"
    )
    assert isinstance(embedding, CreateEmbeddingResponse), f"Embedding has invalid type: {type(embedding)}"
    assert embedding.model == tag, f"Embedding model is incorrect: {embedding.model}"
    assert embedding.usage.total_tokens == 0, f"Embedding usage is non zero: {embedding.usage.total_tokens}"

def test_create_embeddings_openai ():
    openai = OpenAI()
    openai = locally(openai)
    embedding = openai.embeddings.create(
        model="text-embedding-3-large",
        input="What is the capital of France?"
    )
    assert embedding.model == "text-embedding-3-large"