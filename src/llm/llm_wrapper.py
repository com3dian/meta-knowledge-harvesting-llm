import os
from typing import Any, AsyncIterator, Union

from .gemini_llm import gemini_complete_if_cache
from .llm_cache import get_llm_cache

LLM_MODEL_NAME = "gemini-2.5-flash-preview-05-20"


def llm_complete(
    prompt: str,
    history_messages: list[dict[str, Any]] | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.2,
    system_prompt: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    if history_messages is None:
        history_messages = []

    cache = get_llm_cache()

    if api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    return gemini_complete_if_cache(
        model=LLM_MODEL_NAME,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=api_key,
        hashing_kv=cache,
        temperature=temperature,
        max_tokens=max_tokens or 1024,
        **kwargs,
    ) 