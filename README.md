# Paper Check

AI scientific paper review with structure, coherence, and tone checks powered by LangGraph LLM agents. Upload a PDF and the site highlights problematic parts; hover to see concise explanations. Use sidebar filters (Tone, Structure, Coherence) and manage Agents via the Settings modal.

This project includes:

- Backend (FastAPI) with modular components
  - `backend/agents.py`: LangGraph agents (Stylist, Coherence Analyst, Coordinator)
  - `backend/api.py`: API endpoints to analyze a PDF and return highlight rectangles
  - PDF parsing and quote-to-rectangle mapping via PyMuPDF
- Frontend (served from `frontend/`)
  - Modern UI using your provided styles
  - PDF.js rendering with overlay highlights and hover tooltips (Tippy.js)

## Requirements

- Python 3.13
- Environment variables in `.env`:
  - `GOOGLE_API_KEY` (required)

Note: No Java or LanguageTool is required. Only the Google API key is needed for Gemini.

## Install

Use uv (recommended):

```bash
uv sync
```

## Run the backend

```bash
uv run uvicorn backend.app:app --reload --port 8000
```

Backend endpoints:

- `POST /api/analyze_pdf` with multipart/form-data field `file` (PDF)
- The static UI is served at `/` from `frontend/`

Agents management endpoints:

- `GET /api/agents` — list agents
- `POST /api/agents` — create an agent `{ name, prompt }` (category equals name)
- `PUT /api/agents/{id}` — update an agent (category equals name)
- `DELETE /api/agents/{id}` — delete an agent

## Open the web UI

Open <http://127.0.0.1:8000/> in your browser. Use the Upload and Analyze buttons to pick a PDF and run the analysis. The viewer renders pages with PDF.js and overlays highlight rectangles; hover to see the explanation.

In the sidebar:

- Toggle filters for each agent by name (e.g., `Tone`, `Structure`, `Coherence`) to control highlight visibility.
- Click `Settings` to open the Agents modal. You can add, edit, or remove agents (stored in-memory, reset on server restart). Defaults include `Stylist`, `Structure Reviewer`, and `Coherence Analyst`.

If you need CORS from a different origin, the backend currently allows all origins.

## Notes

- Position mapping uses exact quote search on the PDF to place highlights. If a quote occurs multiple times, multiple highlights will appear.
- The frontend uses a scale of 1.5 for mapping PDF point coordinates to screen pixels.
- You can still use `main.py` to generate an annotated PDF directly; the web app is an alternative interactive viewer.
