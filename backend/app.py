import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from .api import router as api_router


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not GOOGLE_API_KEY and not OPENROUTER_API_KEY:
    print("Warning: Neither GOOGLE_API_KEY nor OPENROUTER_API_KEY is set.")


app = FastAPI(title="Paper Check API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_no_cache_header(request: Request, call_next):
    response = await call_next(request)
    # Prevent caching for development
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Include API routes first
app.include_router(api_router)


@app.get("/health")
def health():
    return {"status": "ok"}


# Serve the static frontend last to avoid shadowing routes
app.mount("/", StaticFiles(directory="frontend", html=True), name="ui")
