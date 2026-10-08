"""Helpers shared by OpenAI streaming workflows."""

from __future__ import annotations

import re


def extract_stream_token(chunk, hide_thinking: bool = True) -> str:
    try:
        delta = chunk.choices[0].delta
    except Exception:
        return ""

    content = str(getattr(delta, "content", "") or "")
    reasoning = str(getattr(delta, "reasoning_content", "") or "")
    text = content if hide_thinking else reasoning + content
    return re.sub(r"</?think>", "", text, flags=re.IGNORECASE)

