from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from tools.vector_store import (
    store_movie_analysis,
    retrieve_similar_analyses,
    store_user_preference,
)
from config import CLAUDE_FAST_MODEL, ANTHROPIC_API_KEY
import os

os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

SYSTEM_PROMPT = """You are the Memory Agent for CinemaForensics.

You manage two types of memory:
- SHORT-TERM: conversation history (handled by LangGraph state automatically)
- LONG-TERM: persistent FAISS vector store for analyses and user preferences

Your tasks depend on what you are asked to do:

STORE task: Call store_movie_analysis to save a completed analysis summary
RETRIEVE task: Call retrieve_similar_analyses to find past analyses
PREFERENCE task: Call store_user_preference to save a user preference

Always confirm what action you took and what was stored or retrieved.
Be concise — return a short plain text summary of what you did.
Never make up stored data — only report what the tools return."""


def create_memory_agent():
    llm = ChatAnthropic(
        model=CLAUDE_FAST_MODEL,
        temperature=0,
        max_tokens=1024,
        timeout=30,
        max_retries=3,
    )
    tools = [store_movie_analysis, retrieve_similar_analyses, store_user_preference]
    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
    return agent


def store_analysis(movie_title: str, summary: str, config: dict = None) -> dict:
    """Store completed analysis in long-term memory."""
    if not movie_title or not summary:
        return {"output": "[MEMORY ERROR] Title and summary required.", "error": True}

    agent = create_memory_agent()
    messages = [HumanMessage(
        content=f"Store this analysis for '{movie_title}':\n\n{summary}"
    )]

    try:
        result = agent.invoke({"messages": messages}, config=config or {})
        return {
            "output": result["messages"][-1].content,
            "movie_title": movie_title,
            "error": False,
        }
    except Exception as e:
        return {
            "output": f"[MEMORY ERROR] Store failed: {str(e)}",
            "movie_title": movie_title,
            "error": True,
        }


def retrieve_memory(query: str, config: dict = None) -> dict:
    """Retrieve past analyses from long-term memory."""
    if not query or not query.strip():
        return {"output": "[MEMORY ERROR] Empty query.", "error": True}

    agent = create_memory_agent()
    messages = [HumanMessage(
        content=f"Retrieve past analyses related to: '{query}'"
    )]

    try:
        result = agent.invoke({"messages": messages}, config=config or {})
        return {
            "output": result["messages"][-1].content,
            "query": query,
            "error": False,
        }
    except Exception as e:
        return {
            "output": f"[MEMORY ERROR] Retrieval failed: {str(e)}",
            "query": query,
            "error": True,
        }


def save_preference(key: str, value: str, config: dict = None) -> dict:
    """Save a user preference to long-term memory."""
    if not key or not value:
        return {"output": "[MEMORY ERROR] Key and value required.", "error": True}

    agent = create_memory_agent()
    messages = [HumanMessage(
        content=f"Store user preference — {key}: {value}"
    )]

    try:
        result = agent.invoke({"messages": messages}, config=config or {})
        return {
            "output": result["messages"][-1].content,
            "key": key,
            "value": value,
            "error": False,
        }
    except Exception as e:
        return {
            "output": f"[MEMORY ERROR] Preference save failed: {str(e)}",
            "error": True,
        }
