import requests
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from config import OMDB_API_KEY, MAX_RETRIES, RETRY_WAIT, FACT_AGENT_TIMEOUT


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_fixed(RETRY_WAIT),
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    reraise=True,
)
def _fetch_omdb(params: dict) -> dict:
    response = requests.get(
        "https://www.omdbapi.com/",
        params={**params, "apikey": OMDB_API_KEY},
        timeout=FACT_AGENT_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


@tool
def omdb_lookup(title: str) -> str:
    """
    Fetch movie metadata from OMDB API including plot, cast, director,
    genre, runtime, and IMDB rating. Use this to get factual movie data.
    Input: movie title as a string.
    """
    if not title or not title.strip():
        return "[OMDB ERROR] Empty title provided."

    try:
        data = _fetch_omdb({"t": title.strip(), "plot": "full", "type": "movie"})

        # OMDB returns Response=False for not found
        if data.get("Response") == "False":
            # Fallback: try search
            search = _fetch_omdb({"s": title.strip(), "type": "movie"})
            if search.get("Response") == "True" and search.get("Search"):
                closest = search["Search"][0]["Title"]
                data = _fetch_omdb({"t": closest, "plot": "full", "type": "movie"})
                if data.get("Response") == "False":
                    return f"[OMDB ERROR] Movie '{title}' not found. Closest match: '{closest}'"
            else:
                return f"[OMDB ERROR] Movie '{title}' not found. Check spelling."

        return (
            f"Title: {data.get('Title', title)} ({data.get('Year', '?')})\n"
            f"Genre: {data.get('Genre', 'Unknown')}\n"
            f"Director: {data.get('Director', 'Unknown')}\n"
            f"Actors: {data.get('Actors', 'Unknown')}\n"
            f"Runtime: {data.get('Runtime', 'Unknown')}\n"
            f"IMDB Rating: {data.get('imdbRating', 'N/A')}\n"
            f"Awards: {data.get('Awards', 'N/A')}\n"
            f"Plot: {data.get('Plot', 'No plot available')}"
        )

    except requests.Timeout:
        return "[OMDB ERROR] Request timed out. Proceeding with limited data."
    except requests.ConnectionError:
        return "[OMDB ERROR] Cannot reach OMDB API."
    except requests.HTTPError as e:
        return f"[OMDB ERROR] HTTP error: {str(e)}"
    except Exception as e:
        return f"[OMDB ERROR] Unexpected error: {str(e)}"
