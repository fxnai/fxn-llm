# 
#   Function
#   Copyright © 2024 NatML Inc. All Rights Reserved.
#

from fxn_llm import locally
from openai import OpenAI
from openai.types import CreateEmbeddingResponse

def test_create_embeddings ():
    openai = OpenAI(api_key="fxn")
    openai = locally(openai)
    embedding = openai.embeddings.create(
        model="@nomic/nomic-embed-text-v1.5-quant",
        input="search_query: What is the capital of France?"
    )
    assert isinstance(embedding, CreateEmbeddingResponse), f"Returned embeddings has invalid type: {type(embedding)}"