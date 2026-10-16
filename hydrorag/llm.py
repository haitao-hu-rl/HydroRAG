from __future__ import annotations

from typing import Any

from .config import Settings


def build_chat_model(settings: Settings) -> Any | None:
    """Create a DeepSeek-compatible chat model, or None for offline mode."""

    if not settings.deepseek_api_key:
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
        temperature=settings.temperature,
    )


def message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "".join(parts)
    return str(content)

