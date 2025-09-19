import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from .api import router as api_router


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY is required for LLM analysis")


app = FastAPI(title="Paper Check API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes first
app.include_router(api_router)


@app.get("/health")
def health():
    return {"status": "ok"}


# Serve the static frontend last to avoid shadowing routes
app.mount("/", StaticFiles(directory="frontend", html=True), name="ui")
