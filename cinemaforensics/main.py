import json
import asyncio
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import threading
import queue
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[SERVER] CinemaForensics API starting up...")
    from tools.vector_store import prewarm
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, prewarm)
    print("[SERVER] Ready. Docs at http://localhost:8000/docs")
    yield
    print("[SERVER] Shutting down.")

app = FastAPI(title="CinemaForensics API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")
    @app.get("/")
    async def serve_frontend():
        index = "frontend/index.html"
        if os.path.exists(index):
            return FileResponse(index)
        return {"message": "Frontend not found."}
else:
    @app.get("/")
    async def root():
        return {"message": "CinemaForensics API running.", "docs": "/docs"}

class AnalyzeRequest(BaseModel):
    movie_title: str
    thread_id: Optional[str] = None
    fast_mode: Optional[bool] = False

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    if not req.movie_title or not req.movie_title.strip():
        return {"error": True, "message": "movie_title cannot be empty"}
    import config
    config.FAST_MODE = req.fast_mode or False
    thread_id = req.thread_id or str(uuid.uuid4())
    loop = asyncio.get_event_loop()
    from agents.orchestrator import analyze_movie, analyze_movie_safe
    result = await loop.run_in_executor(None, analyze_movie, req.movie_title.strip(), thread_id)
    return result

@app.websocket("/ws/analyze")
async def ws_analyze(websocket: WebSocket):
    await websocket.accept()
    try:
        raw = await websocket.receive_text()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            await websocket.send_json({"type": "error", "message": "Invalid JSON payload"})
            return
        movie_title = payload.get("movie_title", "").strip()
        fast_mode   = payload.get("fast_mode", False)
        thread_id   = payload.get("thread_id", str(uuid.uuid4()))
        if not movie_title:
            await websocket.send_json({"type": "error", "message": "movie_title is required"})
            return
        import config
        config.FAST_MODE = fast_mode
        await websocket.send_json({"type": "log", "message": f"[ORCHESTRATOR] Starting analysis for '{movie_title}'...", "agent": "orch"})
        log_queue = queue.Queue()
        result_container = {"result": None}
        import builtins
        original_print = builtins.print
        def capturing_print(*args, **kwargs):
            msg = " ".join(str(a) for a in args)
            original_print(*args, **kwargs)
            log_queue.put(("log", msg))
        def run_pipeline():
            builtins.print = capturing_print
            try:
                from agents.orchestrator import analyze_movie, analyze_movie_safe
                result_container["result"] = analyze_movie_safe(movie_title, thread_id)
            except Exception as e:
                result_container["result"] = {"error": True, "message": f"Pipeline error: {str(e)}", "final_report": None}
            finally:
                builtins.print = original_print
                log_queue.put(("done", None))
        threading.Thread(target=run_pipeline, daemon=True).start()
        ping_counter = 0
        while True:
            try:
                try:
                    msg_type, msg_data = log_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.05)
                    # Send ping every 5 seconds to keep connection alive
                    ping_counter += 1
                    if ping_counter >= 100:
                        ping_counter = 0
                        try:
                            await websocket.send_json({"type": "ping"})
                        except Exception:
                            return
                    continue
                if msg_type == "done":
                    break
                if msg_type == "log":
                    await websocket.send_json({"type": "log", "message": msg_data, "agent": _detect_agent(msg_data)})
            except WebSocketDisconnect:
                return
        result = result_container["result"]
        if result and not result.get("error"):
            await websocket.send_json({"type": "result", "data": result["final_report"], "agent_logs": result.get("agent_logs", [])})
        else:
            await websocket.send_json({"type": "error", "message": result.get("message", "Unknown error") if result else "No result"})
        await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass

def _detect_agent(msg: str) -> str:
    u = msg.upper()
    if "[ETHICS]"       in u: return "ethics"
    if "[FACT]"         in u: return "fact"
    if "[DETECTIVE]"    in u: return "detect"
    if "[COMMUNITY]"    in u: return "comm"
    if "[MEMORY]"       in u: return "mem"
    if "[ORCHESTRATOR]" in u: return "orch"
    return "orch"



class FollowupRequest(BaseModel):
    question: str
    thread_id: str

@app.post("/followup")
async def followup(req: FollowupRequest):
    if not req.question or not req.thread_id:
        return {"error": "question and thread_id are required"}

    from agents.orchestrator import get_graph
    import config as cfg
    import os
    os.environ["ANTHROPIC_API_KEY"] = cfg.ANTHROPIC_API_KEY

    graph = get_graph()
    graph_config = {"configurable": {"thread_id": req.thread_id}}

    try:
        state = graph.get_state(graph_config)
        if not state or not state.values:
            return {"error": "No active session found. Analyze a movie first."}

        final_report = state.values.get("final_report", {})
        movie_title  = state.values.get("movie_title", "unknown")

        lines = [
            "Movie: " + str(movie_title),
            "Risk score: " + str(final_report.get("risk_score", "N/A")),
            "Verdict: " + str(final_report.get("verdict_label", "N/A")),
            "Total plot holes: " + str(final_report.get("total_holes", 0)),
            "Detective verdict: " + str(final_report.get("detective_verdict", "")),
            "Community notes: " + str(final_report.get("community_notes", "")),
            "Analysis notes: " + str(final_report.get("analysis_notes", "")),
        ]

        holes = final_report.get("plot_holes", [])
        if holes:
            lines.append("Plot holes found:")
            for h in holes[:6]:
                sev   = str(h.get("severity", "")).upper()
                title = str(h.get("title", ""))
                desc  = str(h.get("description", ""))[:100]
                lines.append("  [" + sev + "] " + title + " - " + desc)

        context = "\n".join(lines)

        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model=cfg.CLAUDE_FAST_MODEL,
            temperature=0.3,
            max_tokens=1024,
            timeout=30,
        )

        prompt = (
            "You are CinemaForensics assistant. The user just completed a movie analysis.\n\n"
            "Session context:\n" + context + "\n\n"
            "Answer the user follow-up question using this context. "
            "Be concise and reference specific plot holes or scores where relevant.\n\n"
            "User question: " + str(req.question)
        )

        response = llm.invoke(prompt)
        return {"answer": response.content, "thread_id": req.thread_id}

    except Exception as e:
        return {"error": "Follow-up failed: " + str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, ws_ping_interval=20, ws_ping_timeout=300, timeout_keep_alive=300)
