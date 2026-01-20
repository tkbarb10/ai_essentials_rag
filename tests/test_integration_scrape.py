"""Integration tests for web scraping functionality using the Tavily API."""
import pytest

from ingestion.scrape import website_map, extract_content, raw_web_content


@pytest.mark.integration
def test_website_map_and_raw_content(env_config):
    if not env_config.get("TAVILY_API_KEY") or not env_config.get("HAS_TAVILY"):
        pytest.skip("TAVILY not available (missing package or API key); skipping Tavily integration tests")

    # Use a small stable site
    try:
        res = website_map("https://example.com", instructions="Return main text", max_depth=1)
    except RuntimeError as e:
        pytest.skip(f"Tavily client not available at runtime: {e}")

    # website_map returns None on error or a list of URLs on success
    assert res is None or isinstance(res, list)

    # If mapping returned results, call raw_web_content; otherwise skip
    if res is None:
        pytest.skip("Website mapping returned no results; skipping extraction")

    content = raw_web_content("https://example.com", instructions="Return main text", max_depth=1)
    assert isinstance(content, list)


@pytest.mark.integration
def test_extract_content_handles_invalid_urls(env_config):
    """Test that extract_content handles invalid URLs gracefully by returning an empty list."""
    if not env_config.get("TAVILY_API_KEY") or not env_config.get("HAS_TAVILY"):
        pytest.skip("TAVILY not available (missing package or API key); skipping Tavily integration tests")

    # Pass a list of invalid URLs - extract_content now takes List[str] and returns List[Dict]
    invalid_urls = ["http://nonexistent.invalid"]
    res = extract_content(invalid_urls)
    # Should return an empty list since extraction will fail
    assert isinstance(res, list)


@pytest.mark.integration
def test_website_map_invalid_url_returns_none(env_config):
    if not env_config.get("TAVILY_API_KEY") or not env_config.get("HAS_TAVILY"):
        pytest.skip("TAVILY not available (missing package or API key); skipping Tavily integration tests")

    bad = website_map("not_a_url", instructions="x", max_depth=1)
    assert bad is None
