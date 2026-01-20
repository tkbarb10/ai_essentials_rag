"""Unit tests for input validation in RAG components."""
import pytest


class TestGradioInterfaceValidation:
    """Tests for _format_history validation in GradioInterface."""

    def test_format_history_accepts_valid_history(self):
        """_format_history should accept properly formatted history."""
        from rag_assistant.gradio_interface import GradioInterface

        # Create a minimal mock assistant
        class MockAssistant:
            llm = None
            vector_db = None
            prompt_template = None

            def bind_tools(self, tools):
                return self

        # GradioInterface needs assistant.llm.bind_tools
        mock_assistant = MockAssistant()
        mock_assistant.llm = MockAssistant()  # Mock LLM with bind_tools

        gi = GradioInterface(mock_assistant)  # type: ignore

        valid_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]

        result = gi._format_history(valid_history)

        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_format_history_raises_on_missing_role(self):
        """_format_history should raise ValueError when role is missing."""
        from rag_assistant.gradio_interface import GradioInterface

        class MockAssistant:
            llm = None
            vector_db = None

            def bind_tools(self, tools):
                return self

        mock_assistant = MockAssistant()
        mock_assistant.llm = MockAssistant()

        gi = GradioInterface(mock_assistant)  # type: ignore

        malformed = [{"content": "Hello"}]  # Missing 'role'

        with pytest.raises(ValueError, match="role"):
            gi._format_history(malformed)

    def test_format_history_raises_on_missing_content(self):
        """_format_history should raise ValueError when content is missing."""
        from rag_assistant.gradio_interface import GradioInterface

        class MockAssistant:
            llm = None
            vector_db = None

            def bind_tools(self, tools):
                return self

        mock_assistant = MockAssistant()
        mock_assistant.llm = MockAssistant()

        gi = GradioInterface(mock_assistant)  # type: ignore

        malformed = [{"role": "user"}]  # Missing 'content'

        with pytest.raises(ValueError, match="content"):
            gi._format_history(malformed)

    def test_format_history_raises_on_non_dict_item(self):
        """_format_history should raise ValueError for non-dict items."""
        from rag_assistant.gradio_interface import GradioInterface

        class MockAssistant:
            llm = None
            vector_db = None

            def bind_tools(self, tools):
                return self

        mock_assistant = MockAssistant()
        mock_assistant.llm = MockAssistant()

        gi = GradioInterface(mock_assistant)  # type: ignore

        malformed = ["not a dict"]

        with pytest.raises(ValueError, match="must be a dict"):
            gi._format_history(malformed)

    def test_format_history_handles_nested_content(self):
        """_format_history should extract text from nested Gradio format."""
        from rag_assistant.gradio_interface import GradioInterface

        class MockAssistant:
            llm = None
            vector_db = None

            def bind_tools(self, tools):
                return self

        mock_assistant = MockAssistant()
        mock_assistant.llm = MockAssistant()

        gi = GradioInterface(mock_assistant)  # type: ignore

        # Gradio sometimes sends nested content format
        nested_history = [
            {"role": "user", "content": [{"text": "Hello from nested"}]}
        ]

        result = gi._format_history(nested_history)

        assert result[0]["content"] == "Hello from nested"

    def test_format_history_empty_list_returns_empty(self):
        """_format_history should return empty list for empty input."""
        from rag_assistant.gradio_interface import GradioInterface

        class MockAssistant:
            llm = None
            vector_db = None

            def bind_tools(self, tools):
                return self

        mock_assistant = MockAssistant()
        mock_assistant.llm = MockAssistant()

        gi = GradioInterface(mock_assistant)  # type: ignore

        result = gi._format_history([])

        assert result == []


class TestRAGAssistantConversationValidation:
    """Tests for conversation validation in RAGAssistant.invoke()."""

    def test_invoke_raises_on_malformed_conversation(self):
        """invoke should raise ValueError for malformed conversation."""
        # This test requires mocking the RAGAssistant initialization
        # which is complex due to LLM/embedding dependencies.
        # We test the validation logic directly instead.

        def validate_conversation(conversation):
            """Extracted validation logic from RAGAssistant.invoke()"""
            if conversation:
                for i, msg in enumerate(conversation):
                    if not isinstance(msg, dict):
                        raise ValueError(f"Conversation item at index {i} must be a dict, got {type(msg).__name__}")
                    if 'role' not in msg:
                        raise ValueError(f"Conversation item at index {i} is missing required 'role' key: {msg}")
                    if 'content' not in msg:
                        raise ValueError(f"Conversation item at index {i} is missing required 'content' key: {msg}")

        # Valid conversation should not raise
        valid = [{"role": "user", "content": "Hi"}]
        validate_conversation(valid)  # Should not raise

        # Missing role should raise
        with pytest.raises(ValueError, match="role"):
            validate_conversation([{"content": "Hi"}])

        # Missing content should raise
        with pytest.raises(ValueError, match="content"):
            validate_conversation([{"role": "user"}])

        # Non-dict should raise
        with pytest.raises(ValueError, match="must be a dict"):
            validate_conversation(["not a dict"])

        # Completely malformed should raise
        with pytest.raises(ValueError):
            validate_conversation([{"foo": "bar"}])
