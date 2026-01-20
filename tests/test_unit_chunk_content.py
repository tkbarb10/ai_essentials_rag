"""Unit tests for utils.chunk_content module."""
import pytest


class TestChunkMarkdownText:
    """Tests for the chunk_markdown_text function."""

    def test_chunk_single_string_returns_list(self):
        """Chunking a single string should return a list of documents."""
        from utils.chunk_content import chunk_markdown_text

        content = "# Header\n\nSome text content here."
        result = chunk_markdown_text(content)

        assert isinstance(result, list)
        assert len(result) >= 1

    def test_chunk_list_of_strings_returns_flat_list(self):
        """Chunking a list of strings should return a flattened list."""
        from utils.chunk_content import chunk_markdown_text

        contents = [
            "# First\n\nFirst content.",
            "# Second\n\nSecond content."
        ]
        result = chunk_markdown_text(contents)

        assert isinstance(result, list)
        assert len(result) >= 2

    def test_chunk_respects_chunk_size_parameter(self):
        """Custom chunk_size should be respected (with real library)."""
        import sys
        from utils.chunk_content import chunk_markdown_text

        # Skip this test when using stubs (stubs don't implement real splitting)
        lts = sys.modules.get('langchain_text_splitters')
        if lts and getattr(lts, '__stub__', False):
            pytest.skip("Skipping chunk_size test with stub implementation")

        # Create content that exceeds chunk_size
        long_content = "# Header\n\n" + "word " * 500
        result = chunk_markdown_text(long_content, chunk_size=100, chunk_overlap=10)

        # Should produce multiple chunks due to small chunk_size
        assert len(result) > 1

    def test_chunk_empty_string_returns_list(self):
        """Chunking empty string should return a list."""
        from utils.chunk_content import chunk_markdown_text

        result = chunk_markdown_text("")

        assert isinstance(result, list)
        # Note: stub may return a document even for empty content;
        # real library returns empty list. Both are acceptable.

    def test_chunk_content_has_page_content_attribute(self):
        """Chunked documents should have page_content attribute."""
        from utils.chunk_content import chunk_markdown_text

        content = "# Test\n\nTest content here."
        result = chunk_markdown_text(content)

        if len(result) > 0:
            assert hasattr(result[0], 'page_content')
            assert isinstance(result[0].page_content, str)

    def test_chunk_content_has_metadata_attribute(self):
        """Chunked documents should have metadata attribute."""
        from utils.chunk_content import chunk_markdown_text

        content = "# Test\n\nTest content here."
        result = chunk_markdown_text(content)

        if len(result) > 0:
            assert hasattr(result[0], 'metadata')
            assert isinstance(result[0].metadata, dict)

    def test_chunk_with_custom_md_kwargs(self):
        """Custom markdown kwargs should be accepted."""
        from utils.chunk_content import chunk_markdown_text

        content = "#### Deep Header\n\nContent under deep header."
        custom_md = {"headers_to_split_on": [("####", "Header 4")]}

        # Should not raise
        result = chunk_markdown_text(content, md_kwargs=custom_md)
        assert isinstance(result, list)

    def test_chunk_with_custom_rc_kwargs(self):
        """Custom recursive character splitter kwargs should be accepted."""
        from utils.chunk_content import chunk_markdown_text

        content = "# Test\n\n" + "Long content. " * 100
        custom_rc = {"separators": ["\n\n", "\n", " "]}

        # Should not raise - must set chunk_overlap < chunk_size
        result = chunk_markdown_text(content, chunk_size=200, chunk_overlap=50, rc_kwargs=custom_rc)
        assert isinstance(result, list)
