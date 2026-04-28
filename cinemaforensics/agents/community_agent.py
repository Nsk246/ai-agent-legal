from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from tools.search_tool import search_plot_holes, search_community_verdict
from config import CLAUDE_FAST_MODEL, ANTHROPIC_API_KEY
import json
import os

os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

SYSTEM_PROMPT = """You are the Community Agent for CinemaForensics.

Your job is to search the web for what real fans and critics say about plot holes in a movie.

Steps:
1. Call search_plot_holes to find fan-reported issues
2. Call search_community_verdict to find overall consensus
3. Analyze the results and return structured JSON

For community_agreement_pct scoring:
- 0-20:  Almost no one discusses plot holes, film considered airtight
- 21-40: Minor discussions, most fans accept the logic
- 41-60: Moderate debate, some well-known issues acknowledged
- 61-80: Strong community consensus that plot holes exist
- 81-100: Overwhelming agreement on significant plot holes

IMPORTANT: Base the score on what the search results actually show.
If multiple Reddit threads and sites discuss plot holes = high score (60+).
If search returns plot hole discussions = there IS community awareness, score accordingly.
Never default to 0 unless search truly returned nothing relevant.

Output ONLY valid JSON — no markdown fences, no text before or after:
{
  "movie_title": "<title>",
  "community_agreement_pct": <int 0-100>,
  "top_community_issues": [
    {
      "title": "<issue title>",
      "severity": "<high|medium|low>",
      "description": "<what the community says>",
      "source": "<reddit/imdb/other>",
      "category": "community"
    }
  ],
  "community_notes": "<overall sentiment summary>",
  "sources_found": <int>
}"""


def create_community_agent():
    llm = ChatAnthropic(
        model=CLAUDE_FAST_MODEL,
        temperature=0,
        max_tokens=2048,
        timeout=40,
        max_retries=3,
    )
    tools = [search_plot_holes, search_community_verdict]
    return create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    cleaned = text.strip()
    if "```" in cleaned:
        lines = cleaned.split("\n")
        cleaned = "\n".join(l for l in lines if not l.strip().startswith("```"))
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Find JSON block by braces
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass
    return None


def run_community_agent(movie_title: str, config: dict = None) -> dict:
    if not movie_title or not movie_title.strip():
        return {
            "output": '{"movie_title":"","community_agreement_pct":0,"top_community_issues":[],"community_notes":"No title","sources_found":0}',
            "movie_title": movie_title,
            "error": True,
        }

    agent = create_community_agent()
    messages = [HumanMessage(
        content=f"Search for community-reported plot holes for the movie: '{movie_title}'. Score the community_agreement_pct based on how much discussion you find."
    )]

    try:
        result = agent.invoke({"messages": messages}, config=config or {})
        final_message = result["messages"][-1].content
        parsed = _extract_json(final_message)

        if parsed:
            parsed.setdefault("movie_title", movie_title)
            parsed.setdefault("community_agreement_pct", 0)
            parsed.setdefault("top_community_issues", [])
            parsed.setdefault("community_notes", "")
            parsed.setdefault("sources_found", 0)
            return {
                "output": json.dumps(parsed),
                "parsed": parsed,
                "movie_title": movie_title,
                "error": False,
            }
        else:
            fallback = {
                "movie_title": movie_title,
                "community_agreement_pct": 0,
                "top_community_issues": [],
                "community_notes": "JSON parsing failed.",
                "sources_found": 0,
            }
            return {"output": json.dumps(fallback), "parsed": fallback, "movie_title": movie_title, "error": True}

    except Exception as e:
        error_msg = f"[COMMUNITY AGENT ERROR] {str(e)}"
        print(error_msg)
        fallback = {
            "movie_title": movie_title,
            "community_agreement_pct": 0,
            "top_community_issues": [],
            "community_notes": error_msg,
            "sources_found": 0,
        }
        return {"output": json.dumps(fallback), "parsed": fallback, "movie_title": movie_title, "error": True}
