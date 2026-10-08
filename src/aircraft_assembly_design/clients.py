"""Lazy construction of external API and database clients."""

from __future__ import annotations

from functools import lru_cache

from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from openai import OpenAI

from .config import SETTINGS


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    return OpenAI()


@lru_cache(maxsize=2)
def get_chat_model(streaming: bool = False) -> ChatOpenAI:
    return ChatOpenAI(
        model=SETTINGS.model,
        temperature=SETTINGS.temperature,
        streaming=streaming,
    )


@lru_cache(maxsize=1)
def get_graph() -> Neo4jGraph:
    return Neo4jGraph()


def clear_client_caches() -> None:
    """Clear cached clients, primarily for tests and configuration reloads."""
    get_openai_client.cache_clear()
    get_chat_model.cache_clear()
    get_graph.cache_clear()

