import pytest

from rag_assistant.rag_assistant import RAGAssistant
from utils.chunk_content import chunk_markdown_text


@pytest.mark.integration
def test_rag_assistant_init_add_query(env_config, test_chroma_dir):
    try:
        assistant = RAGAssistant(persist_path=test_chroma_dir, collection_name="rag_integration", topic="Integration Test")
    except ValueError as e:
        pytest.skip(f"LLM initialization failed: {e}")

    docs = ["# Fruit\n\nApples are red and sweet."]
    chunked = chunk_markdown_text(docs, chunk_size=100, chunk_overlap=5)
    assistant.add_documents(chunked)

    out = assistant.invoke("What color are apples?", conversation=[], n_results=2)
    assert isinstance(out, str)
    assert len(out) > 0


@pytest.mark.integration
def test_rag_assistant_empty_store_returns_string(env_config, test_chroma_dir):
    try:
        assistant = RAGAssistant(persist_path=test_chroma_dir, collection_name="rag_integration_empty", topic="Empty")
    except ValueError as e:
        pytest.skip(f"LLM initialization failed: {e}")

    res = assistant.invoke("Will this crash?", conversation=[], n_results=1)
    assert isinstance(res, str)

