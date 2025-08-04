import os
import logging
from typing import Any, AsyncIterator, Union

from google import genai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import numpy as np

from utils import (
    safe_unicode_decode,
    verbose_debug,
    logger,
    wrap_embedding_func_with_attrs,
    locate_json_string_body_from_string,
)

__api_version__ = "0153"


class InvalidResponseError(Exception):
    """Custom exception class for triggering retry mechanism"""
    pass


def create_gemini_client(model_name: str = "gemini-pro", api_key: str | None = None):
    """Create a Gemini client with the specified model."""
    if not api_key:
        api_key = os.environ["GOOGLE_API_KEY"]

    return genai.Client(api_key=api_key)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((InvalidResponseError,))
)
def gemini_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    api_key: str | None = None,
    token_tracker: Any | None = None,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    """
    Complete a prompt using Gemini API with optional system prompts and message history.

    Args:
        model: Gemini model name (e.g., "gemini-pro").
        prompt: The prompt string.
        system_prompt: Optional context/system instructions (currently merged with user prompt).
        history_messages: Previous conversation messages (optional).
        api_key: Google AI Studio API Key (if not set via env).
        **kwargs: Additional parameters for generation.

    Returns:
        The completion text, or an async iterator (if streaming support added later).
    """

    if history_messages is None:
        history_messages = []

    # Logging
    if logger.level == logging.DEBUG:
        verbose_debug(f"Prompt: {prompt}")
        verbose_debug(f"System prompt: {system_prompt}")
        verbose_debug(f"History: {history_messages}")
    
    if api_key is None:
        api_key = os.environ["GEMINI_API_KEY"]

    client = create_gemini_client(model_name=model, api_key=api_key)

    # Format the full prompt
    full_prompt = ""
    if system_prompt:
        full_prompt += f"{system_prompt}\n\n"
    for message in history_messages:
        full_prompt += f"{message.get('role', 'user')}: {message['content']}\n"
    full_prompt += f"user: {prompt}"

    try:
        response = client.models.generate_content(model=model, contents=full_prompt)
        content = response.text
    except Exception as e:
        logger.error(f"Gemini API Call Failed,\nModel: {model},\nPrompt: {prompt}, Got: {e}")
        raise

    if not content or content.strip() == "":
        logger.error("Received empty content from Gemini API")
        raise InvalidResponseError("Received empty content from Gemini API")

    if r"\u" in content:
        content = safe_unicode_decode(content.encode("utf-8"))

    # Token tracking placeholder (not available from Gemini yet)
    if token_tracker:
        token_tracker.add_usage({
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        })

    return content


@wrap_embedding_func_with_attrs(embedding_dim=768, max_token_size=8192)  # Gemini embeddings are 768-dimensional
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((InvalidResponseError,))
)
async def gemini_embed(
    texts: list[str],
    model: str = "gemini-embedding-exp-03-07",  # Gemini's embedding model
    api_key: str | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Generate embeddings for a list of texts using Gemini's API.

    Args:
        texts: List of texts to embed.
        model: The Gemini embedding model to use.
        api_key: Google AI Studio API Key (if not set via env).
        **kwargs: Additional keyword arguments to pass to the Gemini API.

    Returns:
        A numpy array of embeddings, one per input text.

    Raises:
        InvalidResponseError: If the response from Gemini is invalid or empty.
    """
    if api_key is None:
        api_key = os.environ["GEMINI_API_KEY"]

    client = create_gemini_client(model_name=model, api_key=api_key)

    try:
        embeddings = []
        for text in texts:
            response = client.models.embed_content(model=model, contents=text)
            if not response or not hasattr(response, 'embeddings'):
                raise InvalidResponseError("Invalid response from Gemini API")
            embeddings.append(response.embeddings)
        
        return np.array(embeddings)
    except Exception as e:
        logger.error(f"Gemini Embedding API Call Failed,\nModel: {model},\nTexts: {texts}, Got: {e}")
        raise InvalidResponseError(f"Gemini Embedding API Call Failed: {str(e)}")
