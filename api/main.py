"""Clipt FastAPI application entry point."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router

app = FastAPI(
    title="Clipt API",
    version="1.0.0",
    description=(
        "Thin API layer for the Clipt AI content-repurposing engine. "
        "Provides background job execution for YouTube video analysis and 9:16 vertical processing."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow CORS for future web frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
