"""
VectorForge — FastAPI Application Main Entry Point
"""

from contextlib import asynccontextmanager
import os
from fastapi import FastAPI

from app.api.routes_search import router as search_router
from app.api.routes_system import router as system_router
from app.api.routes_vectors import router as vectors_router
from app.config import DATA_DIR
from app.indexes.manager import IndexManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize IndexManager on startup and auto-load dataset if available."""
    manager = IndexManager()
    if (
        os.path.exists(os.path.join(DATA_DIR, "vectors.npy"))
        and not os.environ.get("VECTORFORGE_SKIP_AUTOLOAD")
    ):
        manager.load_data(DATA_DIR)
    else:
        manager.build_index("brute")
    app.state.manager = manager
    yield


app = FastAPI(
    title="VectorForge API",
    description="Custom Vector Database from Scratch — Exact & Approximate Nearest Neighbor Search",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(system_router)
app.include_router(vectors_router)
app.include_router(search_router)


@app.get("/")
async def root():
    return {"message": "Welcome to VectorForge API", "docs": "/docs"}
