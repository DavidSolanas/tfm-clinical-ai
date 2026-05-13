from typing import Protocol

import anthropic
from openai import OpenAI


class LLMClient(Protocol):
    def generate(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str: ...


class AnthropicClient:
    def __init__(self, client: anthropic.Anthropic, model: str = "claude-sonnet-4-6"):
        self._client = client
        self._model = model

    def generate(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        # System prompt is cached across calls to amortise its token cost.
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text


class OpenAICompatibleClient:
    """Targets any OpenAI-compatible endpoint.

    For llama.cpp server started with `llama-server -m model.gguf --port 8001`,
    use ``base_url='http://localhost:8001/v1'``. The ``model`` field is
    accepted but typically ignored by llama.cpp (it serves the loaded model).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8001/v1",
        model: str = "local",
        api_key: str = "not-needed",
    ):
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    def generate(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content
