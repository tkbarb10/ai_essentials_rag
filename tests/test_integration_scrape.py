import pytest

from ingestion.scrape import website_map, extract_links, extract_content, raw_web_content


@pytest.mark.integration
def test_website_map_and_raw_content(env_config):
    if not env_config.get("TAVILY_API_KEY") or not env_config.get("HAS_TAVILY"):
        pytest.skip("TAVILY not available (missing package or API key); skipping Tavily integration tests")

    # Use a small stable site
    try:
        res = website_map("https://example.com", instructions="Return main text", max_depth=1)
    except RuntimeError as e:
        pytest.skip(f"Tavily client not available at runtime: {e}")

    assert res is None or isinstance(res, dict)

    # If mapping returned results, call raw_web_content; otherwise skip
    if res is None:
        pytest.skip("Website mapping returned no results; skipping extraction")

    content = raw_web_content("https://example.com", instructions="Return main text", max_depth=1)
    assert isinstance(content, list)


@pytest.mark.integration
def test_extract_content_handles_invalid_urls():
    # Pass an invalid URL and ensure extract_content handles errors gracefully
    url_dict = {"group_0": ["http://nonexistent.invalid"]}
    res = extract_content(url_dict) # type: ignore
    assert isinstance(res, dict)
    # The group should be present and its value should be a list (possibly empty)
    assert "group_0" in res
    assert isinstance(res["group_0"], list)


@pytest.mark.integration
def test_website_map_invalid_url_returns_none(env_config):
    if not env_config.get("TAVILY_API_KEY") or not env_config.get("HAS_TAVILY"):
        pytest.skip("TAVILY not available (missing package or API key); skipping Tavily integration tests")

    bad = website_map("not_a_url", instructions="x", max_depth=1)
    assert bad is None
