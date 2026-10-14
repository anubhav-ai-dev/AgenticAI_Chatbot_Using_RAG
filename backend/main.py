"""
FastAPI entry point — the only module at the top level of the package.

All of these start the same app:

    uvicorn backend.main:app --port 8000     # from the project root (production/Procfile)
    python -m backend.main                    # from the project root
    python backend/main.py                    # from anywhere
    python main.py                            # from inside backend/

Package layout (dependencies point downward):

    main.py      app factory, lifespan, static files
    api/         HTTP routes, MCP server              (delivery)
    services/    chat, retrieval, LLM, documents …    (business logic)
    client/      LLM / Tavily / embedding / web I/O   (external systems)
    core/        config, database, exceptions, DI     (application core)
    models/      pydantic schemas + ORM models
    utils/       pure helpers, caches, rate limiter
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    # Started as a plain script: make `backend` importable and keep this directory off
    # sys.path so backend/{utils,client,core,...} are not importable as top-level modules.
    _here = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _here]
    sys.path.insert(0, str(_here.parent))

import logging  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
from typing import Optional  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from backend.api.routes import router  # noqa: E402
from backend.core.config import FRONTEND_DIR, Settings, configure_logging, get_settings  # noqa: E402
from backend.core.container import ServiceContainer  # noqa: E402

logger = logging.getLogger(__name__)


def create_app(settings: Optional[Settings] = None,
               container: Optional[ServiceContainer] = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        c = container or ServiceContainer.build(settings)
        app.state.container = c
        await c.startup()
        logger.info("Application started")
        try:
            yield
        finally:
            await c.shutdown()

    app = FastAPI(
        title="AI Assistant Pro — Agentic RAG Platform",
        description="Low-latency chatbot with hybrid RAG, web search, streaming and free-model fallback.",
        version="4.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    # Frontend (single-page app)
    for name in ("css", "js", "assets"):
        directory = FRONTEND_DIR / name
        if directory.is_dir():
            app.mount(f"/{name}", StaticFiles(directory=str(directory)), name=name)

    @app.get("/", include_in_schema=False)
    def serve_spa():
        index_path = FRONTEND_DIR / "index.html"
        if not index_path.exists():
            return {"message": "Frontend not found. API is running."}
        return FileResponse(str(index_path))

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    s = get_settings()
    uvicorn.run(app, host=s.backend_host, port=s.backend_port)
