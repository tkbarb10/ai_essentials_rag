"""Unit tests for utils.load_yaml_config module."""
import pytest
from pathlib import Path


class TestLoadYamlConfig:
    """Tests for the load_yaml_config function."""

    def test_load_yaml_config_returns_dict(self, tmp_path):
        """load_yaml_config should return a dictionary."""
        from utils.load_yaml_config import load_yaml_config

        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("key: value\nnested:\n  inner: data")

        result = load_yaml_config(yaml_file)

        assert isinstance(result, dict)
        assert result["key"] == "value"
        assert result["nested"]["inner"] == "data"

    def test_load_yaml_config_raises_on_missing_file(self):
        """load_yaml_config should raise FileNotFoundError for missing files."""
        from utils.load_yaml_config import load_yaml_config

        with pytest.raises(FileNotFoundError):
            load_yaml_config("/nonexistent/path/file.yaml")

    def test_load_yaml_config_handles_empty_file(self, tmp_path):
        """load_yaml_config should handle empty files (returns None)."""
        from utils.load_yaml_config import load_yaml_config

        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")

        result = load_yaml_config(yaml_file)

        # yaml.safe_load returns None for empty files
        assert result is None

    def test_load_yaml_config_handles_lists(self, tmp_path):
        """load_yaml_config should handle YAML with lists."""
        from utils.load_yaml_config import load_yaml_config

        yaml_file = tmp_path / "list.yaml"
        yaml_file.write_text("items:\n  - one\n  - two\n  - three")

        result = load_yaml_config(yaml_file)

        assert result["items"] == ["one", "two", "three"]

    def test_load_yaml_config_accepts_string_path(self, tmp_path):
        """load_yaml_config should accept string paths."""
        from utils.load_yaml_config import load_yaml_config

        yaml_file = tmp_path / "string.yaml"
        yaml_file.write_text("test: true")

        result = load_yaml_config(str(yaml_file))

        assert result["test"] is True

    def test_load_yaml_config_accepts_path_object(self, tmp_path):
        """load_yaml_config should accept Path objects."""
        from utils.load_yaml_config import load_yaml_config

        yaml_file = tmp_path / "path.yaml"
        yaml_file.write_text("test: 123")

        result = load_yaml_config(yaml_file)

        assert result["test"] == 123


class TestLoadAllPrompts:
    """Tests for the load_all_prompts function."""

    def test_load_all_prompts_returns_dict(self, tmp_path):
        """load_all_prompts should return a combined dictionary."""
        from utils.load_yaml_config import load_all_prompts

        # Create minimal YAML files
        (tmp_path / "rag_prompts.yaml").write_text("qa_assistant:\n  instructions: Answer")
        (tmp_path / "ingestion_prompts.yaml").write_text("prep_prompt:\n  instructions: Prepare")
        (tmp_path / "components.yaml").write_text("tones:\n  conversational: Be friendly")

        result = load_all_prompts(tmp_path)

        assert isinstance(result, dict)
        assert "qa_assistant" in result
        assert "prep_prompt" in result
        assert "tones" in result

    def test_load_all_prompts_raises_on_missing_dir(self):
        """load_all_prompts should raise FileNotFoundError for missing directory."""
        from utils.load_yaml_config import load_all_prompts

        with pytest.raises(FileNotFoundError):
            load_all_prompts("/nonexistent/prompts/dir")

    def test_load_all_prompts_handles_missing_optional_files(self, tmp_path):
        """load_all_prompts should handle missing optional files gracefully."""
        from utils.load_yaml_config import load_all_prompts

        # Only create one file
        (tmp_path / "rag_prompts.yaml").write_text("test_prompt:\n  instructions: Test")

        # Should not raise even if other files are missing
        result = load_all_prompts(tmp_path)

        assert isinstance(result, dict)
        assert "test_prompt" in result

    def test_load_all_prompts_merges_configs(self, tmp_path):
        """load_all_prompts should merge all configurations."""
        from utils.load_yaml_config import load_all_prompts

        (tmp_path / "rag_prompts.yaml").write_text("key1: value1")
        (tmp_path / "ingestion_prompts.yaml").write_text("key2: value2")
        (tmp_path / "components.yaml").write_text("key3: value3")

        result = load_all_prompts(tmp_path)

        assert result.get("key1") == "value1"
        assert result.get("key2") == "value2"
        assert result.get("key3") == "value3"


class TestLoadAllPromptsFromRealDirectory:
    """Tests that load actual prompt files from the project."""

    def test_load_prompts_from_project_prompts_dir(self):
        """load_all_prompts should successfully load from actual prompts directory."""
        from utils.load_yaml_config import load_all_prompts
        from config.paths import PROMPTS_DIR

        if not Path(PROMPTS_DIR).exists():
            pytest.skip("Prompts directory not found")

        result = load_all_prompts(PROMPTS_DIR)

        assert isinstance(result, dict)
        # Should have at least some content
        assert len(result) > 0
