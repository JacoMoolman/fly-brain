from __future__ import annotations

import argparse
import asyncio
import threading
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from flygym import assets_dir
from pydantic import BaseModel

from .paths import ProjectPaths
from .runtime import IntegratedRuntime

MESH_DIR = assets_dir / "model/neuromechfly/meshes/simplified_max2000faces"


class ControlRequest(BaseModel):
    paused: bool | None = None
    assist: float | None = None
    stimulus: str | None = None
    reset: bool = False
    wing_stimulus: bool | None = None


paths = ProjectPaths.discover()
runtime: IntegratedRuntime | None = None
_morphology = None
_morphology_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global runtime
    runtime = IntegratedRuntime()
    runtime.start()
    yield
    runtime.stop()


app = FastAPI(title="MaleCNS and FlyGym", lifespan=lifespan)
gui_dir = paths.project / "gui"
app.mount("/meshes", StaticFiles(directory=MESH_DIR), name="meshes")
app.mount("/static", StaticFiles(directory=gui_dir), name="static")


@app.get("/")
def index():
    return FileResponse(gui_dir / "index.html")


@app.get("/api/health")
def health():
    return {"ready": runtime is not None, "status": runtime.state()["status"] if runtime else "loading"}


@app.get("/api/layout")
def layout():
    if runtime is None:
        raise HTTPException(503, "Runtime is loading")
    payload = runtime.layout()
    if payload is None:
        raise HTTPException(503, "Brain graph is still loading")
    return payload


@app.get("/api/skeleton")
def skeleton():
    if runtime is None:
        raise HTTPException(503, "Runtime is loading")
    payload = runtime.skeleton()
    if payload is None:
        raise HTTPException(503, "Fly body is still loading")
    return payload


@app.get("/api/morphology")
def morphology(offset: int = Query(0, ge=0), count: int = Query(32, ge=1, le=64),
               detail: int = Query(1024, ge=32, le=4096), body_id: int | None = None):
    global _morphology
    if runtime is None or not runtime.ready:
        raise HTTPException(503, "Brain graph is still loading")
    with _morphology_lock:
        if _morphology is None:
            from .morphology import Morphology
            _morphology = Morphology(paths, runtime.brain)
    try:
        data, attempted, failures = _morphology.batch(offset, count, detail, body_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return Response(data, media_type="application/octet-stream", headers={
        "X-Neurons-Attempted": str(attempted), "X-Neurons-Missing": str(len(failures)),
        "X-Neurons-Total": str(runtime.brain.n),
    })


@app.get("/api/state")
def state():
    if runtime is None:
        raise HTTPException(503, "Runtime is loading")
    return runtime.state()


@app.websocket("/ws/state")
async def stream_state(websocket: WebSocket):
    """Push each new simulation frame as soon as it exists, instead of polling."""
    await websocket.accept()
    last_key = None
    try:
        while True:
            if runtime is not None:
                snapshot = runtime.state()
                iteration = snapshot.get("loop", {}).get("iteration", -1)
                key = (iteration, snapshot.get("status"), str(snapshot.get("control")), snapshot.get("error"))
                if key != last_key:
                    last_key = key
                    await websocket.send_json(snapshot)
            # Waiting on receive (not sleep) lets server shutdown close this loop even when no frames change.
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=0.015)
            except asyncio.TimeoutError:
                continue
            if message["type"] == "websocket.disconnect":
                return
    except (WebSocketDisconnect, RuntimeError, OSError):
        return


@app.post("/api/control")
def control(request: ControlRequest):
    if runtime is None:
        raise HTTPException(503, "Runtime is loading")
    try:
        return runtime.control(**request.model_dump())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local MaleCNS and FlyGym viewer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    uvicorn.run("flybrain_local.server:app", host=args.host, port=args.port, reload=False,
                timeout_graceful_shutdown=3)


if __name__ == "__main__":
    main()
