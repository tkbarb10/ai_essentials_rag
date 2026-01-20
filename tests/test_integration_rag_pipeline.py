import pytest
from types import SimpleNamespace

from rag_assistant.rag_assistant import RAGAssistant
from utils.chunk_content import chunk_markdown_text


@pytest.mark.integration
def test_full_rag_pipeline(env_config, test_chroma_dir):
    # Attempt to initialize RAGAssistant; skip if LLM cannot be initialized
    try:
        assistant = RAGAssistant(persist_path=test_chroma_dir, collection_name="integration_test", topic="Integration")
    except ValueError as e:
        pytest.skip(f"LLM initialization failed: {e}")

    docs = ["# Fruit\n\nApples are red and sweet.", "# Vegetables\n\nCarrots are orange."]
    chunked = chunk_markdown_text(docs, chunk_size=100, chunk_overlap=10)
    assistant.add_documents(chunked)

    # Query a question that should be answerable from the docs
    res = assistant.invoke("What color are apples?", conversation=[], n_results=2)
    assert isinstance(res, str)
    assert len(res) > 0


@pytest.mark.integration
def test_empty_vector_store_does_not_crash(env_config, test_chroma_dir):
    try:
        assistant = RAGAssistant(persist_path=test_chroma_dir, collection_name="integration_empty", topic="Empty")
    except ValueError as e:
        pytest.skip(f"LLM initialization failed: {e}")

    # Do not add documents
    out = assistant.invoke("Who am I?", conversation=[], n_results=2)
    assert isinstance(out, str)


@pytest.mark.integration
def test_malformed_history_handled(env_config, test_chroma_dir):
    try:
        assistant = RAGAssistant(persist_path=test_chroma_dir, collection_name="integration_malformed", topic="Malformed")
    except ValueError as e:
        pytest.skip(f"LLM initialization failed: {e}")

    docs = ["# Test\n\nSample text about testing."]
    assistant.add_documents(chunk_markdown_text(docs))

    # Malformed history: the message coercion layer should raise a clear ValueError
    malformed = [{"foo": "bar"}]
    with pytest.raises(ValueError):
        assistant.invoke("Tell me about testing.", conversation=malformed, n_results=1)
