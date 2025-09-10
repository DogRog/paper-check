import os
import io
import json
from typing import List, Dict, Any

import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from tempfile import NamedTemporaryFile

from langchain_google_genai import ChatGoogleGenerativeAI
from main import run_analysis_to_annotations  # reuse langgraph pipeline
import language_tool_python

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY is required for LLM analysis")

# Initialize local LanguageTool (requires Java installed)
LT_LANG = os.getenv("LT_LANG", "en-US")
lt_tool = language_tool_python.LanguageTool(LT_LANG)

app = FastAPI(title="Paper Check API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the web UI
app.mount("/web", StaticFiles(directory="web", html=True), name="web")


class Annotation(BaseModel):
    page: int
    rect: List[float]  # [x1, y1, x2, y2]
    agent: str
    quote: str
    comment: str


def lt_check(text: str):
    return lt_tool.check(text)


def build_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash")


def extract_pages_text(doc: fitz.Document) -> List[str]:
    return [page.get_text() for page in doc]


def find_quote_instances(doc: fitz.Document, quote: str) -> List[Dict[str, Any]]:
    results = []
    if not quote or not quote.strip():
        return results
    q = quote.strip()
    for page_index, page in enumerate(doc):
        rects = page.search_for(q)
        for r in rects:
            results.append({
                "page": page_index,
                "rect": [r.x0, r.y0, r.x1, r.y1],
            })
    return results


def _expand_to_word_boundaries(text: str, offset: int, length: int) -> str:
    """Expand a match to full word boundaries; if very short, include context words.

    - If length >= 3: return the substring expanded to word boundaries.
    - If resulting token is < 3 chars (e.g., 'a'), include prev and next words to form a short phrase.
    """
    n = len(text)
    if n == 0:
        return ""
    start = max(0, min(offset, n-1))
    end = max(start, min(offset + max(0, length), n))

    # Expand left
    while start > 0 and (text[start-1].isalnum() or text[start-1] in "-_”’'"):
        start -= 1
    # Expand right
    while end < n and (text[end].isalnum() or text[end] in "-_”’'"):
        end += 1
    token = text[start:end].strip()

    # If token too short, build phrase with neighbors
    def grab_prev_word(i: int) -> str:
        j = i-1
        while j > 0 and not text[j].isalnum():
            j -= 1
        k = j
        while k >= 0 and (text[k].isalnum() or text[k] in "-_”’'"):
            k -= 1
        return text[k+1:j+1].strip()

    def grab_next_word(i: int) -> str:
        j = i
        while j < n and not text[j].isalnum():
            j += 1
        k = j
        while k < n and (text[k].isalnum() or text[k] in "-_”’'"):
            k += 1
        return text[j:k].strip()

    if len(token) < 3:
        prev_w = grab_prev_word(start)
        next_w = grab_next_word(end)
        phrase = " ".join([w for w in [prev_w, token, next_w] if w])
        token = phrase.strip() or token

    # Trim very long tokens to avoid excessive search length
    if len(token) > 120:
        token = token[:120]
    return token


async def style_and_coherence_feedback(full_text: str) -> List[Dict[str, str]]:
    llm = build_llm()
    json_suffix = (
        """
Your response MUST be a valid JSON list of objects. Each object should have two keys: 'quote' and 'comment'.
- 'quote': The exact, verbatim text snippet from the document that your feedback pertains to.
- 'comment': Your analysis or suggestion regarding that specific quote.
If you find no issues, return an empty list: [].
"""
    )
    prompts = [
        ("Stylist", "You are a scientific style editor. Assess clarity, conciseness, academic tone." + json_suffix),
        ("Coherence Analyst", "You are a logical coherence analyst. Evaluate structure and flow, look for gaps or contradictions." + json_suffix),
    ]
    feedback: List[Dict[str, str]] = []
    for agent_name, system_prompt in prompts:
        msg = llm.invoke(f"{system_prompt}\n\nText:\n'''\n{full_text}\n'''")
        content = msg.content if hasattr(msg, "content") else str(msg)
        try:
            # Extract JSON list
            import re
            m = re.search(r"```json\s*([\s\S]*?)\s*```|(\[[\s\S]*\])", content)
            s = m.group(1) if m and m.group(1) else (m.group(2) if m else "[]")
            items = json.loads(s)
            for it in items:
                feedback.append({
                    "agent": agent_name,
                    "quote": (it.get("quote") or "").strip(),
                    "comment": (it.get("comment") or "").strip(),
                })
        except Exception:
            pass
    return feedback


async def grammar_feedback_with_positions(doc: fitz.Document, full_text: str) -> List[Dict[str, Any]]:
    matches = lt_check(full_text)
    results: List[Dict[str, Any]] = []
    for m in matches:
        try:
            offset = m.offset
            length = m.errorLength or 0
            _ = full_text[offset: offset + length] if length > 0 else (m.shortMessage or "")
            quote = _expand_to_word_boundaries(full_text, offset, length)
            comment = m.message or (getattr(m, "ruleIssueType", None) or "Grammar")
            quote = (quote or "").strip()
            if not quote:
                continue
            for inst in find_quote_instances(doc, quote):
                results.append({
                    "agent": "Grammar (LanguageTool)",
                    "quote": quote,
                    "comment": comment,
                    "page": inst["page"],
                    "rect": inst["rect"],
                })
        except Exception:
            continue
    return results


async def unify_feedback_with_positions(doc: fitz.Document, full_text: str, input_pdf_path: str) -> List[Annotation]:
    out: List[Annotation] = []
    # Grammar
    grammar = await grammar_feedback_with_positions(doc, full_text)
    # Style + Coherence via langgraph (reuse main.py pipeline)
    try:
        langgraph_items: List[Dict[str, Any]] = run_analysis_to_annotations(input_pdf_path)
    except Exception:
        langgraph_items = []
    # Keep only style/coherence (drop proofreader to avoid duplication with grammar API)
    for it in langgraph_items:
        agent = it.get("agent", "")
        if agent not in ("Stylist", "Coherence Analyst"):
            continue
        out.append(Annotation(
            page=int(it.get("page", 0)),
            rect=[float(x) for x in it.get("rect", [0, 0, 0, 0])],
            agent=agent,
            quote=it.get("quote", ""),
            comment=it.get("comment", ""),
        ))
    # Add grammar items (already with positions)
    for g in grammar:
        out.append(Annotation(
            page=g["page"],
            rect=g["rect"],
            agent=g["agent"],
            quote=g["quote"],
            comment=g["comment"],
        ))
    return out


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file upload")
    # Save temporarily to reuse main.run_analysis_to_annotations which expects a path
    with NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        doc = fitz.open(stream=io.BytesIO(data), filetype="pdf")
        full_text = "".join(page.get_text() for page in doc)
        annotations = await unify_feedback_with_positions(doc, full_text, tmp_path)
        doc.close()
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return {"annotations": [a.model_dump() for a in annotations]}


@app.get("/")
def root():
    return {"status": "ok"}
