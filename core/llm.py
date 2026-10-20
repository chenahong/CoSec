"""OpenAI-compatible LLM helper for DynaComSec.

The benchmark only needs chat-completions style calls for the attacker,
scenario generation, and optional judging. Keeping this local removes the
runtime dependency on the larger Prism package.
"""

from __future__ import annotations

import os
import time
from typing import Any


DEFAULT_MODELS = {
    "openai": "gpt-4o",
}


def _ensure_messages(prompt: str | list[dict[str, str]]) -> list[dict[str, str]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return prompt


def generate(
    prompt: str | list[dict[str, str]],
    llm_type: str = "openai",
    model: str | None = None,
    temperature: float = 0.7,
    response_format: str | None = None,
    max_retries: int = 3,
) -> str:
    """Generate text using an OpenAI-compatible chat-completions endpoint."""

    if llm_type != "openai":
        raise ValueError(
            f"Unsupported DynaComSec LLM provider: {llm_type}. "
            "Use llm_type='openai' with OPENAI_BASE_URL for compatible proxies."
        )

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install the openai package to use LLM attacker/generation.") from exc

    client_kwargs: dict[str, Any] = {"api_key": os.environ.get("OPENAI_API_KEY", "")}
    if os.environ.get("OPENAI_BASE_URL"):
        client_kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
    client = OpenAI(**client_kwargs)

    request: dict[str, Any] = {
        "model": model or os.environ.get("OPENAI_MODEL") or DEFAULT_MODELS["openai"],
        "messages": _ensure_messages(prompt),
    }
    if not request["model"].startswith(("gpt-5", "o3", "o4")):
        request["temperature"] = temperature
    if response_format == "json":
        request["response_format"] = {"type": "json_object"}

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**request)
            return response.choices[0].message.content or ""
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(min(2**attempt * 2, 10))

    return ""
