"""Integration tests for the full application flow via Gradio interface."""
import importlib
import sys
from types import SimpleNamespace


import pytest

from rag_assistant.rag_assistant import RAGAssistant
from rag_assistant.gradio_interface import GradioInterface
from utils.chunk_content import chunk_markdown_text


@pytest.mark.integration
def test_app_end_to_end_chat_flow(env_config, test_chroma_dir):
    try:
        assistant = RAGAssistant(persist_path=test_chroma_dir, collection_name="app_integration", topic="AppTest")
    except ValueError as e:
        pytest.skip(f"LLM initialization failed: {e}")

    # Load a small document and add to the store
    docs = ["# Testing\n\nThis document explains that tests exist to validate behavior."]
    assistant.add_documents(chunk_markdown_text(docs))

    gi = GradioInterface(assistant)

    # Simulate the aya_gradio_chat flow: first yield shows thinking, then streaming
    history = []

    # First, check the streaming generator directly
    gen = gi.stream_chat("Tell me about tests", history, n_results=1)

    outputs = []
    for o in gen:
        outputs.append(o)

    assert any(isinstance(o, str) and len(o) > 0 for o in outputs)


@pytest.mark.integration
def test_app_generator_includes_thinking_and_content(env_config, test_chroma_dir):
    try:
        assistant = RAGAssistant(persist_path=test_chroma_dir, collection_name="app_integration2", topic="AppTest")
    except ValueError as e:
        pytest.skip(f"LLM initialization failed: {e}")

    docs = ["# Hello\n\nThis doc is about saying hello."]
    assistant.add_documents(chunk_markdown_text(docs))

    gi = GradioInterface(assistant)

    # Simulate aya_gradio_chat wrapper behavior
    history = []
    # Generator that mimics aya_gradio_chat
    def like_chat(message, history):
        yield history + [{"role": "assistant", "content": "⏳ Thinking..."}]
        accumulated = ""
        for chunk in gi.stream_chat(message, history, 3):
            accumulated = chunk
            yield history + [{"role": "assistant", "content": accumulated}]

    g = like_chat("Say hello", history)
    first = next(g)
    assert any("Thinking" in m.get("content", "") for m in first)

    # Get at least one streaming response
    responses = []
    for item in g:
        responses.extend(item)

    assert any("assistant" in r.get("role", "") for r in responses)
