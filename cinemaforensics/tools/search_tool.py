import os
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from config import TAVILY_API_KEY

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

_tavily = TavilySearch(max_results=5, search_depth="advanced")


def _extract_results(raw) -> list[dict]:
    """
    Handle both dict (new Tavily) and list (old Tavily) response formats.
    Always returns a clean list of {title, url, content} dicts.
    """
    if isinstance(raw, dict):
        return raw.get("results", [])
    elif isinstance(raw, list):
        return raw
    return []


def _format_results(results: list[dict], cap: int = 5) -> str:
    """Format search results into readable string for LLM."""
    seen_urls = set()
    formatted = []
    for r in results:
        url = r.get("url", "")
        if url in seen_urls:
            continue
        seen_urls.add(url)
        formatted.append(
            f"SOURCE: {r.get('title', 'Unknown')}\n"
            f"URL: {url}\n"
            f"CONTENT: {str(r.get('content', ''))[:600]}\n"
        )
        if len(formatted) >= cap:
            break
    return formatted


@tool
def search_plot_holes(movie_title: str) -> str:
    """
    Search Reddit, IMDb forums, and film analysis sites for fan-reported
    plot holes, continuity errors, and logical inconsistencies in a movie.
    Input: movie title as a string.
    """
    if not movie_title or not movie_title.strip():
        return "[SEARCH ERROR] Empty movie title provided."

    queries = [
        f"{movie_title} plot holes reddit",
        f"{movie_title} movie continuity errors logical inconsistencies",
        f"{movie_title} film mistakes analysis",
    ]

    all_results = []
    seen_urls = set()

    for query in queries:
        try:
            raw = _tavily.invoke({"query": query})
            results = _extract_results(raw)
            for r in results:
                url = r.get("url", "")
                if url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(r)
        except Exception as e:
            all_results.append({"title": "Search error", "url": "", "content": f"Query failed: {str(e)}"})
            continue

    if not all_results:
        return f"[SEARCH ERROR] No results found for '{movie_title}'."

    formatted = _format_results(all_results, cap=6)
    return (
        f"Community search results for '{movie_title}' ({len(formatted)} sources):\n\n"
        + "\n---\n".join(formatted)
    )


@tool
def search_community_verdict(movie_title: str) -> str:
    """
    Search for overall community consensus, critical analysis, and
    debate about a movie's plot logic and narrative consistency.
    Input: movie title as a string.
    """
    if not movie_title or not movie_title.strip():
        return "[SEARCH ERROR] Empty movie title provided."

    queries = [
        f"{movie_title} plot analysis fan verdict discussion",
        f"{movie_title} movie plot consistency review",
    ]

    all_results = []
    seen_urls = set()

    for query in queries:
        try:
            raw = _tavily.invoke({"query": query})
            results = _extract_results(raw)
            for r in results:
                url = r.get("url", "")
                if url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(r)
        except Exception as e:
            continue

    if not all_results:
        return f"[SEARCH ERROR] No community verdict found for '{movie_title}'."

    formatted = _format_results(all_results, cap=4)
    return (
        f"Community verdict for '{movie_title}' ({len(formatted)} sources):\n\n"
        + "\n---\n".join(formatted)
    )
