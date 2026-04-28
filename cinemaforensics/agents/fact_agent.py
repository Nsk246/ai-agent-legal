from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent
from tools.omdb_tool import omdb_lookup
from tools.wiki_tool import wiki_lookup
from config import CLAUDE_FAST_MODEL, ANTHROPIC_API_KEY
import os

os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

SYSTEM_PROMPT = """You are the Fact Agent for CinemaForensics — a movie intelligence system.

Your ONLY job is to gather raw factual data about a movie. You do NOT analyze or judge.

You must:
1. Call omdb_lookup to get movie metadata (plot, cast, director, genre, runtime, rating)
2. Call wiki_lookup to get the full Wikipedia article (deeper plot, production details)
3. Return ALL gathered data in a clean structured format

Output format — always return exactly this structure:
MOVIE_TITLE: <title>
YEAR: <year>
GENRE: <genre>
DIRECTOR: <director>
RUNTIME: <runtime>
IMDB_RATING: <rating>
CAST: <main actors>
PLOT_SUMMARY: <full plot from OMDB>
WIKI_CONTENT: <full wiki content>
ERRORS: <list any tools that failed, or NONE>

Rules:
- If OMDB fails, still try Wikipedia and vice versa
- If BOTH fail, return what you know and list errors clearly
- Never make up or hallucinate movie data
- Never skip a tool call — always try both"""


def create_fact_agent():
    llm = ChatAnthropic(
        model=CLAUDE_FAST_MODEL,
        temperature=0,
        max_tokens=2048,
        timeout=30,
        max_retries=3,
    )
    tools = [omdb_lookup, wiki_lookup]
    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
    return agent


def run_fact_agent(movie_title: str, config: dict = None) -> dict:
    """
    Run the fact agent for a given movie title.
    Returns dict with content and any errors.
    """
    if not movie_title or not movie_title.strip():
        return {
            "output": "[FACT AGENT ERROR] Empty movie title.",
            "movie_title": movie_title,
            "error": True,
        }

    agent = create_fact_agent()
    messages = [HumanMessage(content=f"Gather all factual data for the movie: '{movie_title}'")]

    try:
        result = agent.invoke(
            {"messages": messages},
            config=config or {},
        )

        # Extract final AI message
        final_message = result["messages"][-1].content

        return {
            "output": final_message,
            "movie_title": movie_title,
            "error": False,
        }

    except Exception as e:
        error_msg = f"[FACT AGENT ERROR] Agent execution failed: {str(e)}"
        print(error_msg)
        return {
            "output": error_msg,
            "movie_title": movie_title,
            "error": True,
        }
