# Paper Check

Evaluate scientific papers for grammatical, stylistic, and coherence issues.

This project provides:

- Backend (FastAPI) that analyzes PDFs
	- Grammar via language_tool_python (local LanguageTool)
	- Style & Coherence via Gemini (langchain-google-genai) in a simple langgraph-like sequence
- Web UI that renders the PDF with PDF.js and shows hoverable pop-up explanations using Tippy.js

## Requirements

- Python 3.13
- Environment variables in `.env`:

	- `GOOGLE_API_KEY` (required)

Additionally, `language_tool_python` requires Java (JRE 8+) installed and on PATH.

## Install

Use uv (recommended):

```bash
uv sync
```

## Run the backend

```bash
uv run uvicorn server:app --reload --port 8000
```

Backend endpoints:

- `GET /` health
- `POST /analyze` with multipart/form-data field `file` (PDF)

## Open the web UI

Open `web/index.html` in your browser. Use the form to upload a PDF; the UI renders pages with PDF.js and overlays highlights. Hover a highlight to see the Tippy tooltip with details.

If you need CORS from a different origin, the backend currently allows all origins.

## Notes

- Position mapping uses exact quote search on the PDF to place highlights. If a quote occurs multiple times, multiple highlights will appear.
- Coordinates assume a rendering scale of 1.5 in the frontend, matching the overlay placement logic.
- You can still use `main.py` to generate an annotated PDF directly; the web app is an alternative interactive viewer.
