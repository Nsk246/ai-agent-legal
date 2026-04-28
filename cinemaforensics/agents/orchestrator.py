import json
import operator
from typing import Annotated, TypedDict, Literal
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from agents.fact_agent import run_fact_agent
from agents.detective_agent import run_detective_agent
from agents.community_agent import run_community_agent
from agents.memory_agent import store_analysis, retrieve_memory
from config import CLAUDE_REASONING_MODEL, ANTHROPIC_API_KEY
import os

os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY


# ── State schema ──────────────────────────────────────────────────────────────
# Keys written by multiple parallel nodes MUST use Annotated + operator.add
class GraphState(TypedDict):
    movie_title:        str
    user_message:       str
    fact_data:          str
    detective_data:     str
    community_data:     str
    memory_context:     str
    detective_parsed:   dict | None
    community_parsed:   dict | None
    final_report:       dict | None

    # These use operator.add so parallel nodes can safely append
    errors:             Annotated[list[str], operator.add]
    agent_logs:         Annotated[list[str], operator.add]
    messages:           Annotated[list[BaseMessage], operator.add]

    # current_step written by multiple nodes — use last-writer-wins via list
    current_steps:      Annotated[list[str], operator.add]


# ── Helper ────────────────────────────────────────────────────────────────────
def _safe_parse_json(text: str, fallback: dict) -> dict:
    if not text:
        return fallback
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1])
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return fallback


# ── Node 1: Memory Retrieval ──────────────────────────────────────────────────
def node_retrieve_memory(state: GraphState) -> dict:
    movie_title = state["movie_title"]
    log = f"[MEMORY] Checking long-term memory for '{movie_title}'..."
    print(log)

    result = retrieve_memory(f"movie analysis {movie_title}")

    return {
        "memory_context": result["output"],
        "current_steps":  ["memory_retrieved"],
        "agent_logs":     [log, f"[MEMORY] {result['output'][:120]}"],
    }


# ── Node 2: Fact Gathering ────────────────────────────────────────────────────
def node_gather_facts(state: GraphState) -> dict:
    movie_title = state["movie_title"]
    log = f"[FACT] Fetching data for '{movie_title}' via OMDB + Wikipedia..."
    print(log)

    result = run_fact_agent(movie_title)

    errors = []
    if result["error"]:
        errors.append(f"[FACT ERROR] {result['output'][:100]}")

    return {
        "fact_data":     result["output"],
        "current_steps": ["facts_gathered"],
        "errors":        errors,
        "agent_logs":    [log, f"[FACT] Gathered {len(result['output'])} chars of data."],
    }


# ── Node 3: Detective Analysis ────────────────────────────────────────────────
def node_detective_analysis(state: GraphState) -> dict:
    movie_title = state["movie_title"]
    fact_data   = state.get("fact_data", "")
    log = f"[DETECTIVE] Analyzing plot logic for '{movie_title}'..."
    print(log)

    if not fact_data or "[FACT AGENT ERROR]" in fact_data:
        fact_data = f"Movie title: {movie_title}. Limited external data available."

    result = run_detective_agent(movie_title, fact_data)

    errors = []
    if result["error"]:
        errors.append(f"[DETECTIVE ERROR] {result.get('error_detail', result['output'][:100])}")

    parsed = result.get("parsed") or _safe_parse_json(result["output"], {
        "movie_title":    movie_title,
        "risk_score":     0.0,
        "verdict":        "Analysis unavailable",
        "plot_holes":     [],
        "analysis_notes": "Detective agent encountered an error.",
    })

    hole_count = len(parsed.get("plot_holes", []))
    return {
        "detective_data":   result["output"],
        "detective_parsed": parsed,
        "current_steps":    ["detective_done"],
        "errors":           errors,
        "agent_logs": [
            log,
            f"[DETECTIVE] Found {hole_count} plot holes. Risk score: {parsed.get('risk_score', 'N/A')}",
        ],
    }


# ── Node 4: Community Research ────────────────────────────────────────────────
def node_community_research(state: GraphState) -> dict:
    movie_title = state["movie_title"]
    log = f"[COMMUNITY] Searching fan discussions for '{movie_title}'..."
    print(log)

    result = run_community_agent(movie_title)

    errors = []
    if result["error"]:
        errors.append(f"[COMMUNITY ERROR] {result['output'][:100]}")

    parsed = result.get("parsed") or _safe_parse_json(result["output"], {
        "movie_title":             movie_title,
        "community_agreement_pct": 0,
        "top_community_issues":    [],
        "community_notes":         "Community search unavailable.",
        "sources_found":           0,
    })

    return {
        "community_data":   result["output"],
        "community_parsed": parsed,
        "current_steps":    ["community_done"],
        "errors":           errors,
        "agent_logs": [
            log,
            f"[COMMUNITY] Agreement: {parsed.get('community_agreement_pct', 0)}% | Sources: {parsed.get('sources_found', 0)}",
        ],
    }


# ── Node 5: Synthesize ────────────────────────────────────────────────────────
def node_synthesize(state: GraphState) -> dict:
    movie_title      = state["movie_title"]
    detective_parsed = state.get("detective_parsed") or {}
    community_parsed = state.get("community_parsed") or {}
    errors           = state.get("errors", [])
    log = f"[ORCHESTRATOR] Synthesizing final report for '{movie_title}'..."
    print(log)

    # Merge + deduplicate plot holes
    all_holes  = []
    seen_titles = set()

    for hole in detective_parsed.get("plot_holes", []):
        t = hole.get("title", "").lower()
        if t and t not in seen_titles:
            seen_titles.add(t)
            all_holes.append(hole)

    for hole in community_parsed.get("top_community_issues", []):
        t = hole.get("title", "").lower()
        if t and t not in seen_titles:
            seen_titles.add(t)
            hole["source"] = hole.get("source", "community")
            all_holes.append(hole)

    # Sort by severity
    severity_order = {"high": 0, "medium": 1, "low": 2}
    all_holes.sort(key=lambda h: severity_order.get(h.get("severity", "low"), 2))

    # Weighted risk score
    detective_score  = float(detective_parsed.get("risk_score", 0))
    community_score  = community_parsed.get("community_agreement_pct", 0) / 10.0
    final_risk_score = round((detective_score * 0.7) + (community_score * 0.3), 1)

    high_count   = sum(1 for h in all_holes if h.get("severity") == "high")
    medium_count = sum(1 for h in all_holes if h.get("severity") == "medium")
    low_count    = sum(1 for h in all_holes if h.get("severity") == "low")

    if final_risk_score >= 7:
        verdict_label = "High Plot Risk"
    elif final_risk_score >= 4:
        verdict_label = "Moderate Risk"
    else:
        verdict_label = "Low Risk"

    final_report = {
        "movie_title":         movie_title,
        "verdict_label":       verdict_label,
        "risk_score":          final_risk_score,
        "community_agreement": f"{community_parsed.get('community_agreement_pct', 0)}%",
        "total_holes":         len(all_holes),
        "severity_breakdown":  {"high": high_count, "medium": medium_count, "low": low_count},
        "plot_holes":          all_holes,
        "detective_verdict":   detective_parsed.get("verdict", ""),
        "community_notes":     community_parsed.get("community_notes", ""),
        "analysis_notes":      detective_parsed.get("analysis_notes", ""),
        "errors_encountered":  errors,
        "agent_logs":          state.get("agent_logs", []),
    }

    return {
        "final_report":  final_report,
        "current_steps": ["synthesized"],
        "agent_logs":    [log, f"[ORCHESTRATOR] Report ready — {len(all_holes)} total issues found."],
        "messages":      [AIMessage(content=json.dumps(final_report))],
    }


# ── Node 6: Store Memory ──────────────────────────────────────────────────────
def node_store_memory(state: GraphState) -> dict:
    movie_title  = state["movie_title"]
    final_report = state.get("final_report") or {}
    log = f"[MEMORY] Storing analysis for '{movie_title}' to long-term memory..."
    print(log)

    summary = (
        f"Movie: {movie_title} | "
        f"Risk Score: {final_report.get('risk_score', 0)} | "
        f"Verdict: {final_report.get('verdict_label', '')} | "
        f"Total Issues: {final_report.get('total_holes', 0)} | "
        f"Holes: {json.dumps(final_report.get('plot_holes', [])[:3])}"
    )

    result = store_analysis(movie_title, summary)

    return {
        "current_steps": ["complete"],
        "agent_logs":    [log, f"[MEMORY] {result['output'][:80]}"],
    }


# ── Edge: fan-out condition ───────────────────────────────────────────────────
def fan_out_after_facts(state: GraphState):
    fact_data = state.get("fact_data", "")
    if fact_data and len(fact_data) > 50:
        return ["detective", "community"]
    print("[ORCHESTRATOR] Fact gathering failed — skipping to synthesis.")
    return ["synthesize"]


# ── Build graph ───────────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("retrieve_memory", node_retrieve_memory)
    graph.add_node("gather_facts",    node_gather_facts)
    graph.add_node("detective",       node_detective_analysis)
    graph.add_node("community",       node_community_research)
    graph.add_node("synthesize",      node_synthesize)
    graph.add_node("store_memory",    node_store_memory)

    graph.add_edge(START,             "retrieve_memory")
    graph.add_edge("retrieve_memory", "gather_facts")

    # Fan-out: detective + community run in parallel
    graph.add_conditional_edges(
        "gather_facts",
        fan_out_after_facts,
        ["detective", "community", "synthesize"],
    )

    # Fan-in: both feed into synthesize
    graph.add_edge("detective",    "synthesize")
    graph.add_edge("community",    "synthesize")
    graph.add_edge("synthesize",   "store_memory")
    graph.add_edge("store_memory", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ── Singleton ─────────────────────────────────────────────────────────────────
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ── Public entry point ────────────────────────────────────────────────────────
def analyze_movie(movie_title: str, thread_id: str = "default") -> dict:
    if not movie_title or not movie_title.strip():
        return {"error": True, "message": "Movie title cannot be empty.", "final_report": None}

    graph = get_graph()

    initial_state = {
        "movie_title":      movie_title.strip(),
        "user_message":     f"Analyze plot holes in '{movie_title}'",
        "fact_data":        "",
        "detective_data":   "",
        "community_data":   "",
        "memory_context":   "",
        "detective_parsed": None,
        "community_parsed": None,
        "errors":           [],
        "agent_logs":       [f"[ORCHESTRATOR] Starting analysis for '{movie_title}'"],
        "current_steps":    ["starting"],
        "final_report":     None,
        "messages":         [HumanMessage(content=f"Analyze: {movie_title}")],
    }

    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = graph.invoke(initial_state, config=config)
        report = result.get("final_report")

        if not report:
            return {
                "error":      True,
                "message":    "Pipeline completed but no report generated.",
                "final_report": None,
                "agent_logs": result.get("agent_logs", []),
            }

        return {
            "error":        False,
            "final_report": report,
            "agent_logs":   result.get("agent_logs", []),
        }

    except Exception as e:
        error_msg = f"[ORCHESTRATOR ERROR] Pipeline failed: {str(e)}"
        print(error_msg)
        return {
            "error":        True,
            "message":      error_msg,
            "final_report": None,
            "agent_logs":   [error_msg],
        }


# ── Safe public entry point with ethics check ─────────────────────────────────
def analyze_movie_safe(movie_title: str, thread_id: str = "default") -> dict:
    """
    Runs ethics check first, then the full pipeline.
    Uses the cleaned/corrected title from the ethics agent.
    """
    from agents.ethics_agent import check_ethics

    if not movie_title or not movie_title.strip():
        return {"error": True, "message": "Movie title cannot be empty.", "final_report": None}

    print(f"[ETHICS] Checking input: '{movie_title}'")
    check = check_ethics(movie_title)

    if not check["safe"]:
        print(f"[ETHICS] Blocked: {check['reason']}")
        return {
            "error":        True,
            "flagged":      True,
            "message":      f"⚠ Request blocked: {check['reason']} CinemaForensics only analyzes published films.",
            "final_report": None,
            "agent_logs":   [f"[ETHICS] Blocked — {check['reason']}"],
        }

    # Use the cleaned/corrected title
    clean_title = check.get("suggested_title") or movie_title.strip()
    if clean_title != movie_title.strip():
        print(f"[ETHICS] Title corrected: '{movie_title}' → '{clean_title}'")

    return analyze_movie(clean_title, thread_id)
