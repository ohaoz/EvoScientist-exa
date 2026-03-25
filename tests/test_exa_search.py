import pytest

from EvoScientist import EvoScientist as agent_module
from EvoScientist.tools import search


def test_has_search_api_key_accepts_exa(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("EXA_API_KEY", "exa-test-key")

    assert agent_module._has_search_api_key() is True


@pytest.mark.asyncio
async def test_run_search_uses_exa_when_configured(monkeypatch):
    async def fake_search(query: str, max_results: int, topic: str) -> dict:
        assert query == "llm agents"
        assert max_results == 2
        assert topic == "news"
        return {
            "results": [
                {
                    "title": "Agents paper",
                    "url": "https://example.com/agents",
                    "publishedDate": "2026-03-01T00:00:00.000Z",
                    "author": "Example Author",
                    "text": "Agents are useful.",
                    "highlights": ["Key highlight"],
                    "summary": "Summary text.",
                }
            ]
        }

    monkeypatch.setenv("EXA_API_KEY", "exa-test-key")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(search, "_search_with_exa", fake_search)

    result = await search._run_search("llm agents", max_results=2, topic="news")

    assert "Found 1 result(s) for 'llm agents':" in result
    assert "## Agents paper" in result
    assert "**Highlights:**" in result
    assert "Key highlight" in result
    assert "Agents are useful." in result


@pytest.mark.asyncio
async def test_run_search_reports_missing_provider(monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    result = await search._run_search("llm agents", max_results=1, topic="general")

    assert "no search provider configured" in result.lower()
