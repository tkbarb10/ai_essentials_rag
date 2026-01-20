"""Integration tests for the GradioInterface class."""
import pytest

from rag_assistant.rag_assistant import RAGAssistant
from rag_assistant.gradio_interface import GradioInterface
from utils.chunk_content import chunk_markdown_text


@pytest.mark.integration
def test_gradio_interface_streams_chunks(env_config, test_chroma_dir):
    try:
        assistant = RAGAssistant(persist_path=test_chroma_dir, collection_name="gradio_integration", topic="Gradio")
    except ValueError as e:
        pytest.skip(f"LLM initialization failed: {e}")

    docs = ["# Greeting\n\nHello world from Gradio tests."]
    assistant.add_documents(chunk_markdown_text(docs))

    gi = GradioInterface(assistant)

    outputs = []
    for out in gi.stream_chat("Tell me a greeting", [], n_results=1):
        outputs.append(out)

    # Should produce at least one incremental output and be a string
    assert any(isinstance(o, str) and len(o) > 0 for o in outputs)


@pytest.mark.integration
def test_gradio_interface_with_malformed_history(env_config, test_chroma_dir):
    try:
        assistant = RAGAssistant(persist_path=test_chroma_dir, collection_name="gradio_malformed", topic="Gradio")
    except ValueError as e:
        pytest.skip(f"LLM initialization failed: {e}")

    docs = ["# Topic\n\nSmall piece of content to be retrieved."]
    assistant.add_documents(chunk_markdown_text(docs))

    gi = GradioInterface(assistant)

    # Malformed history should raise a clear error from the message coercion layer
    malformed = [{"foo": "bar"}]
    with pytest.raises(ValueError):
        list(gi.stream_chat("Tell me about the topic", malformed, n_results=1))
