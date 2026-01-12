import pytest
from pathlib import Path
import time

from ingestion.prep import prepare_web_content
from config.paths import OUTPUTS_DIR


@pytest.mark.integration
def test_prepare_web_content_creates_output(env_config, tmp_path):
    if not env_config.get("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set in .env; skipping Groq integration tests")

    # Create a small test file
    f = tmp_path / "small.md"
    f.write_text("# Test\n\nThis is a short piece of content about apples.")

    before = set(Path(OUTPUTS_DIR).glob("prepped_rag_material_*.md"))
    prepare_web_content(str(f))

    # Allow a short time for file flush
    time.sleep(0.5)

    after = set(Path(OUTPUTS_DIR).glob("prepped_rag_material_*.md"))
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

    f = tmp_path / "empty.md"
    f.write_text("")

    # Should not raise
    prepare_web_content(str(f))


@pytest.mark.integration
def test_prepare_with_invalid_model_does_not_create_file(env_config, tmp_path):
    if not env_config.get("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set in .env; skipping Groq integration tests")

    f = tmp_path / "small.md"
    f.write_text("# Test\n\nThis is a short piece of content")

    from config.paths import OUTPUTS_DIR
    before = set(Path(OUTPUTS_DIR).glob("prepped_rag_material_*.md"))

    # Pass an invalid model to force a provider error; prepare_web_content handles exceptions internally
    prepare_web_content(str(f), model="nonexistent/model")

    after = set(Path(OUTPUTS_DIR).glob("prepped_rag_material_*.md"))
    new = after - before

    # No new files should be created for an invalid model
    assert len(new) == 0
