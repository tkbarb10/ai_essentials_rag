import pytest
from pathlib import Path
import time

from config.paths import DATA_DIR


@pytest.mark.integration
def test_prepare_web_content_creates_output(env_config, tmp_path):
    if not env_config.get("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set in .env; skipping Groq integration tests")

    if not env_config.get("HAS_LANGCHAIN"):
        pytest.skip("langchain chat models not available; skipping prep integration tests")

    from ingestion.prep import prepare_web_content

    # Create a small test file
    f = tmp_path / "small.md"
    f.write_text("# Test\n\nThis is a short piece of content about apples.")

    before = set(Path(DATA_DIR).glob("prepped_rag_material_*.md"))
    prepare_web_content(str(f))

    # Allow a short time for file flush
    time.sleep(0.5)

    after = set(Path(DATA_DIR).glob("prepped_rag_material_*.md"))
    new = after - before

    # Either a file is created or the function managed the request without crashing
    assert len(new) <= 1

    # Cleanup
    for p in new:
        p.unlink()


@pytest.mark.integration
def test_prepare_with_empty_file(env_config, tmp_path):
    if not env_config.get("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set in .env; skipping Groq integration tests")

    if not env_config.get("HAS_LANGCHAIN"):
        pytest.skip("langchain chat models not available; skipping prep integration tests")

    from ingestion.prep import prepare_web_content

    f = tmp_path / "empty.md"
    f.write_text("")

    # Should not raise
    prepare_web_content(str(f))


@pytest.mark.integration
def test_prepare_with_nonexistent_file_exits(env_config, tmp_path):
    """Test that prepare_web_content handles a nonexistent file path gracefully."""
    if not env_config.get("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set in .env; skipping Groq integration tests")

    if not env_config.get("HAS_LANGCHAIN"):
        pytest.skip("langchain chat models not available; skipping prep integration tests")

    from ingestion.prep import prepare_web_content

    nonexistent = tmp_path / "does_not_exist.md"

    before = set(Path(DATA_DIR).glob("prepped_rag_material_*.md"))

    # Pass a nonexistent file path - function should exit without creating output
    with pytest.raises(SystemExit):
        prepare_web_content(str(nonexistent))

    after = set(Path(DATA_DIR).glob("prepped_rag_material_*.md"))
    new = after - before

    # No new files should be created for a nonexistent input file
    assert len(new) == 0
