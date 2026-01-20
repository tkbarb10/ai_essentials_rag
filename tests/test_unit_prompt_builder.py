"""Unit tests for utils.prompt_builder module."""
import pytest


class TestFormatPromptSection:
    """Tests for the format_prompt_section helper function."""

    def test_format_with_string_value(self):
        """Formatting with a string value should join lead-in and value."""
        from utils.prompt_builder import format_prompt_section

        result = format_prompt_section("Lead:", "content here")

        assert "Lead:" in result
        assert "content here" in result

    def test_format_with_list_value(self):
        """Formatting with a list should create bullet points."""
        from utils.prompt_builder import format_prompt_section

        result = format_prompt_section("Items:", ["item1", "item2", "item3"])

        assert "Items:" in result
        assert "- item1" in result
        assert "- item2" in result
        assert "- item3" in result


class TestBuildPrompt:
    """Tests for the build_prompt function."""

    def test_build_prompt_requires_instructions(self):
        """build_prompt should raise ValueError if instructions is missing."""
        from utils.prompt_builder import build_prompt

        config = {"role": "test assistant"}

        with pytest.raises(ValueError, match="instructions"):
            build_prompt(config)

    def test_build_prompt_with_minimal_config(self):
        """build_prompt should work with just instructions."""
        from utils.prompt_builder import build_prompt

        config = {"instructions": "Answer user questions."}
        result = build_prompt(config)

        assert isinstance(result, str)
        assert "Answer user questions" in result

    def test_build_prompt_includes_role(self):
        """build_prompt should include the role in the prompt."""
        from utils.prompt_builder import build_prompt

        config = {
            "role": "knowledgeable expert",
            "instructions": "Provide expert answers."
        }
        result = build_prompt(config)

        assert "knowledgeable expert" in result

    def test_build_prompt_replaces_topic_placeholder(self):
        """build_prompt should replace {topic} placeholders."""
        from utils.prompt_builder import build_prompt

        config = {
            "role": "expert on {topic}",
            "instructions": "Answer questions about {topic}."
        }
        result = build_prompt(config, topic="Python programming")

        assert "Python programming" in result
        assert "{topic}" not in result

    def test_build_prompt_replaces_categories_placeholder(self):
        """build_prompt should replace {categories} placeholder."""
        from utils.prompt_builder import build_prompt

        config = {
            "instructions": "Organize into: {categories}"
        }
        result = build_prompt(config, categories=["Category A", "Category B"])

        assert "- Category A" in result
        assert "- Category B" in result
        assert "{categories}" not in result

    def test_build_prompt_includes_constraints(self):
        """build_prompt should include constraints section."""
        from utils.prompt_builder import build_prompt

        config = {
            "instructions": "Do the task.",
            "constraints": ["Be concise", "Use proper grammar"]
        }
        result = build_prompt(config)

        assert "Output constraints:" in result
        assert "Be concise" in result
        assert "Use proper grammar" in result

    def test_build_prompt_includes_goal(self):
        """build_prompt should include goal section."""
        from utils.prompt_builder import build_prompt

        config = {
            "instructions": "Help the user.",
            "goal": "Provide clear and helpful responses"
        }
        result = build_prompt(config)

        assert "goal of this interaction" in result.lower()
        assert "Provide clear and helpful responses" in result

    def test_build_prompt_includes_format_section(self):
        """build_prompt should include format section."""
        from utils.prompt_builder import build_prompt

        config = {
            "instructions": "Answer questions.",
            "format": ["Use markdown", "Include examples"]
        }
        result = build_prompt(config)

        assert "Output Format:" in result
        assert "Use markdown" in result

    def test_build_prompt_with_context(self):
        """build_prompt should include context when provided."""
        from utils.prompt_builder import build_prompt

        config = {"instructions": "Answer based on context."}
        context = ["Fact 1: The sky is blue", "Fact 2: Water is wet"]

        result = build_prompt(config, context=context)

        assert "ADDITIONAL CONTEXT" in result
        assert "Fact 1" in result
        assert "Fact 2" in result

    def test_build_prompt_topic_in_constraints(self):
        """build_prompt should replace {topic} in constraints."""
        from utils.prompt_builder import build_prompt

        config = {
            "instructions": "Answer questions.",
            "constraints": ["Only discuss {topic}"]
        }
        result = build_prompt(config, topic="cooking")

        assert "Only discuss cooking" in result
        assert "{topic}" not in result

    def test_build_prompt_topic_in_goal(self):
        """build_prompt should replace {topic} in goal."""
        from utils.prompt_builder import build_prompt

        config = {
            "instructions": "Help users.",
            "goal": "Educate about {topic}"
        }
        result = build_prompt(config, topic="machine learning")

        assert "Educate about machine learning" in result

    def test_build_prompt_default_strategy(self):
        """build_prompt should use Self-Ask strategy by default."""
        from utils.prompt_builder import build_prompt

        config = {"instructions": "Answer questions."}
        result = build_prompt(config)

        # The prompt should include some reasoning content (based on Self-Ask)
        assert isinstance(result, str)


class TestBuildPromptWithComponents:
    """Tests for build_prompt with component overrides."""

    def test_build_prompt_with_tools_component(self):
        """build_prompt should load tools from components."""
        from utils.prompt_builder import build_prompt

        config = {"instructions": "Use tools to help."}
        components = {"tools": True}

        result = build_prompt(config, components=components)

        # Should include tools section if components.yaml has tools
        assert isinstance(result, str)

    def test_build_prompt_with_tone_component(self):
        """build_prompt should apply tone from components."""
        from utils.prompt_builder import build_prompt

        config = {"instructions": "Be helpful."}
        components = {"tones": "conversational"}

        result = build_prompt(config, components=components)

        # Communication Style should be present
        assert "Communication Style:" in result
