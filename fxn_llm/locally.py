# 
#   Function
#   Copyright © 2024 NatML Inc. All Rights Reserved.
#

from fxn import Function
from numpy import float32
from numpy.typing import NDArray
from re import match
from types import MethodType
from typing import List, Literal, Optional, TypeVar

LLMProvider = Literal["openai", "anthropic"]
LLMClient = TypeVar("LLMClient")

def locally (
    client: LLMClient,
    provider: LLMProvider="openai",
    access_key: str=None,
    api_url: str=None,
    fxn: Function=None
) -> LLMClient:
    """
    Patch your LLM client to run locally.

    Parameters:
        client (OpenAI | Anthropic): LLM provider client.
        provider (LLMProvider): LLM provider identifier. Defaults to `openai`.
        access_key (str): Function access key.
        api_url (str): Function API URL.
        fxn (Function): Function client.

    Returns:
        OpenAI | Anthropic: LLM provider client patched to run locally.
    """
    fxn = fxn if fxn is not None else Function(access_key=access_key, api_url=api_url)
    if provider == "openai":
        from openai.types import CreateEmbeddingResponse, Embedding
        from openai.types.create_embedding_response import Usage
        embeddings_create_openai = client.embeddings.create
        def embeddings_create_fxn (
            self,
            *,
            input: str | List[str],
            model: str,
            **kwargs
        ) -> CreateEmbeddingResponse:
            # Check
            if not match(r"^@[a-z0-9._-]+/[a-z0-9._-]+$", model):
                return embeddings_create_openai(
                    input=input,
                    model=model,
                    **kwargs
                )
            # Check inputs
            assert kwargs.get("dimensions", None) is None, "Explicit dimensionality is not supported"
            assert kwargs.get("encoding_format", "float") == "float", "Base64 encoding format is not yet supported"
            # Predict
            input = [input] if isinstance(input, str) else input
            prediction = fxn.predictions.create(tag=model, inputs={ "input": input })
            embeddings: NDArray[float32] = prediction.results[0]
            # Return
            return CreateEmbeddingResponse(
                data=[Embedding(
                    embedding=data.tolist(),
                    index=idx,
                    object="embedding"
                ) for idx, data in enumerate(embeddings)],
                model=model,
                object="list",
                usage=Usage(prompt_tokens=0, total_tokens=0)
            )
        client.embeddings.create = MethodType(embeddings_create_fxn, client.embeddings)
        return client
    elif provider == "anthropic":
        pass
    return client