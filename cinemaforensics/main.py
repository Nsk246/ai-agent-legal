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

@app.get("/health")
async def health():
    return {"status": "ok", "service": "CinemaForensics"}

@app.get("/memory/search")
async def search_memory(q: str):
    if not q:
        return {"error": "Query parameter q is required"}
    from agents.memory_agent import retrieve_memory
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, retrieve_memory, q)
    return result

class PreferenceRequest(BaseModel):
    key: str
    value: str

@app.post("/memory/preference")
async def save_preference_endpoint(req: PreferenceRequest):
    if not req.key or not req.value:
        return {"error": "Both key and value are required"}
    from agents.memory_agent import save_preference
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, save_preference, req.key, req.value)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, ws_ping_interval=20, ws_ping_timeout=300, timeout_keep_alive=300)
