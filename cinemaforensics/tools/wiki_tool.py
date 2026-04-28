import wikipediaapi
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_fixed
from config import MAX_RETRIES, RETRY_WAIT


WIKI = wikipediaapi.Wikipedia(
    language="en",
    user_agent="CinemaForensics/1.0 (academic project)",
)


def _truncate(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_period = cut.rfind(".")
    if last_period > max_chars * 0.8:
        return cut[:last_period + 1] + "\n[...truncated]"
    return cut + "\n[...truncated]"


@tool
def wiki_lookup(title: str) -> str:
    """
    Fetch Wikipedia article for a movie including full plot summary,
    production details, and critical reception. Use this to get deep
    narrative and story context about a film.
    Input: movie title as a string.
    """
    if not title or not title.strip():
        return "[WIKI ERROR] Empty title provided."

    variants = [
        title.strip(),
        f"{title.strip()} (film)",
        f"{title.strip()} (movie)",
    ]

    for variant in variants:
        try:
            page = WIKI.page(variant)
            if page.exists():
                return (
                    f"Wikipedia: {page.title}\n"
                    f"URL: {page.fullurl}\n\n"
                    f"Summary:\n{_truncate(page.summary, 1500)}\n\n"
                    f"Full Content:\n{_truncate(page.text, 4000)}"
                )
        except Exception:
            continue

    return (
        f"[WIKI ERROR] No Wikipedia article found for '{title}'. "
        f"Tried variants: {variants}. Proceeding with OMDB data only."
    )
