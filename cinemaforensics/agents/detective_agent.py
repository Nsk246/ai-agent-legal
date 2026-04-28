from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from config import CLAUDE_REASONING_MODEL, CLAUDE_FAST_MODEL, ANTHROPIC_API_KEY, FAST_MODE
import json
import time
import os

os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

SYSTEM_PROMPT = """You are the Detective Agent for CinemaForensics — an expert film analyst
specializing in identifying plot holes, continuity errors, and logical inconsistencies.

You receive raw movie data. Your job is PURE REASONING — no tool calls needed.

Analyze the movie data and identify plot holes by checking:
1. TIMELINE CONSISTENCY — do events happen in a logical order? Any paradoxes?
2. CHARACTER LOGIC — do characters act consistently with their established traits?
3. PHYSICS & WORLD RULES — does the movie's internal logic hold up?
4. CAUSE & EFFECT — do actions have believable consequences?
5. CONTINUITY — are there contradictions between scenes?

Output ONLY valid JSON — no markdown, no preamble, no text before or after:
{
  "movie_title": "<title>",
  "risk_score": <float 0.0-10.0>,
  "verdict": "<one sentence overall verdict>",
  "plot_holes": [
    {
      "title": "<short descriptive title>",
      "severity": "<high|medium|low>",
      "description": "<2-3 sentence explanation>",
      "category": "<timeline|character|physics|continuity|cause_effect>",
      "source": "detective reasoning"
    }
  ],
  "analysis_notes": "<broader observations>"
}

Severity: high=breaks core premise, medium=noticeable flaw, low=minor nitpick
Score 0-3: tight narrative | 4-6: noticeable issues | 7-10: fundamental failures
If data is incomplete, analyze what you have and note it in analysis_notes."""


def _extract_json(text: str) -> dict | None:
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
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass
    return None


def _make_fallback(movie_title: str, reason: str) -> dict:
    return {
        "movie_title":    movie_title,
        "risk_score":     0.0,
        "verdict":        "Analysis unavailable",
        "plot_holes":     [],
        "analysis_notes": reason,
    }


def _run_with_model(model: str, fact_data: str, movie_title: str, config: dict) -> dict:
    """Try running detective agent with a specific model."""
    llm = ChatAnthropic(
        model=model,
        temperature=0.2,
        max_tokens=3000,
        timeout=60,
    )
    agent = create_react_agent(llm, [], prompt=SYSTEM_PROMPT)
    prompt = f"Analyze this movie data and identify all plot holes. Return JSON only.\n\n{fact_data}"
    result = agent.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    return result["messages"][-1].content


def run_detective_agent(movie_title: str, fact_data: str, config: dict = None) -> dict:
    if not fact_data or not fact_data.strip():
        fallback = _make_fallback(movie_title, "No fact data provided.")
        return {"output": json.dumps(fallback), "parsed": fallback, "movie_title": movie_title, "error": True}

    config = config or {}

    # Model priority: use reasoning model unless fast mode, fallback to haiku on overload
    models_to_try = (
        [CLAUDE_FAST_MODEL]
        if FAST_MODE
        else [CLAUDE_REASONING_MODEL, CLAUDE_FAST_MODEL]
    )

    last_error = None

    for attempt, model in enumerate(models_to_try):
        try:
            if attempt > 0:
                print(f"[DETECTIVE] Retrying with fallback model: {model}")
                time.sleep(2)  # brief pause before retry

            print(f"[DETECTIVE] Using model: {model}")
            final_message = _run_with_model(model, fact_data, movie_title, config)
            parsed = _extract_json(final_message)

            if parsed:
                parsed.setdefault("movie_title",    movie_title)
                parsed.setdefault("risk_score",     0.0)
                parsed.setdefault("plot_holes",     [])
                parsed.setdefault("verdict",        "Analysis complete")
                parsed.setdefault("analysis_notes", "")
                return {
                    "output":       json.dumps(parsed),
                    "parsed":       parsed,
                    "movie_title":  movie_title,
                    "error":        False,
                }
            else:
                # JSON parse failed — try next model
                last_error = "JSON parse failed"
                continue

        except Exception as e:
            err_str = str(e)
            last_error = err_str

            # Overloaded or rate limited — try next model
            if any(code in err_str for code in ["529", "overloaded", "529", "rate_limit", "529"]):
                print(f"[DETECTIVE] Model {model} overloaded — trying fallback...")
                continue

            # Timeout — try next model
            if "timeout" in err_str.lower():
                print(f"[DETECTIVE] Model {model} timed out — trying fallback...")
                continue

            # Auth or other hard error — don't retry
            if any(code in err_str for code in ["401", "403", "invalid_api_key"]):
                print(f"[DETECTIVE] Auth error — check API key")
                break

            # Unknown error — try next model anyway
            print(f"[DETECTIVE] Error with {model}: {err_str[:100]}")
            continue

    # All models failed
    print(f"[DETECTIVE] All models failed. Last error: {last_error}")
    fallback = _make_fallback(movie_title, f"All models failed: {last_error}")
    return {
        "output":      json.dumps(fallback),
        "parsed":      fallback,
        "movie_title": movie_title,
        "error":       True,
    }
