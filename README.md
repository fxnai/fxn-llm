# Function LLM for Python

![function logo](https://raw.githubusercontent.com/fxnai/.github/main/logo_wide.png)

[![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2Fy5vwgXkz2f%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&logo=discord&logoColor=white&label=Function%20community)](https://fxn.ai/community)
[![X (formerly Twitter) Follow](https://img.shields.io/twitter/follow/fxnai)](https://twitter.com/fxnai)

Use local LLMs in your Python apps, with GPU acceleration and zero dependencies. This package is designed to patch `OpenAI` and `Anthropic` clients for running inference locally, using predictors hosted on [Function](https://fxn.ai/explore).

> [!TIP]
> We offer a similar package for use in the browser and Node.js. Check out [fxn-llm-js](https://github.com/fxnai/fxn-llm-js).

> [!IMPORTANT]
> This package is still a work-in-progress, so the API could change drastically between **all** releases.

## Installing Function LLM
Function is distributed on PyPi. To install, open a terminal and run the following command:
```bash
# Install Function LLM
$ pip install --upgrade fxn-llm
```

> [!NOTE]
> Function LLM requires Python 3.10+

> [!IMPORTANT]
> Make sure to create an access key by signing onto [Function](https://fxn.ai/settings/developer). You'll need it to fetch the predictor at runtime.

## Using the OpenAI Client Locally
To run text generation and embedding models locally using the OpenAI client, patch your `OpenAI` instance with the `locally` function:
```py
from openai import OpenAI
from fxn_llm import locally

# 💥 Create your OpenAI client
openai = OpenAI()

# 🔥 Make it local
openai = locally(openai)

# 🚀 Generate embeddings
embeddings = openai.embeddings.create(
    model="@nomic/nomic-embed-text-v1.5-quant",
    input="search_query: Hello world!"
)
```

> [!WARNING]
> Currently, only `openai.embeddings.create` is supported. Text generation is coming soon!

## Using the Anthropic Client Locally
To run text generation models locally using the Anthopic client, patch your `Anthropic` instance with the `locally` function and the following configuration:
```py
from anthropic import Anthropic
from fxn_llm import locally

# 💥 Create your Anthropic client
anthropic = Anthropic()

# 🔥 Make it local
anthropic = locally(openai, provider="anthropic")

# 🚀 Chat
message = anthropic.messages.create(
  model="@meta/llama-3.1-8b-quant",
  messages=[{ "role": "user", "content": "Hello, Llama" }],
  max_tokens=1024,
)
```

> [!CAUTION]
> Anthropic support is not functional. It is still a work-in-progress.

___

## Useful Links
- [Discover predictors to use in your apps](https://fxn.ai/explore).
- [Join our Discord community](https://fxn.ai/community).
- [Check out our docs](https://docs.fxn.ai).
- Learn more about us [on our blog](https://blog.fxn.ai).
- Reach out to us at [hi@fxn.ai](mailto:hi@fxn.ai).

Function is a product of [NatML Inc](https://github.com/natmlx).