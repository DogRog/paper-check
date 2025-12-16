import os
import json
import tempfile
from typing import List, Dict, Any
import fitz  # pymupdf
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel

from .agents import review_paper

router = APIRouter()

class AnalysisResponse(BaseModel):
    final_score: float | None
    scoring_summary: str | None
    annotations: List[Dict[str, Any]]
    page_sizes: List[Dict[str, Any]]
    statistics: Dict[str, Any]

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract full text from PDF for the LLM"""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def find_quote_coordinates(pdf_path: str, quote: str, location_hint: str = None) -> Dict[str, Any]:
    """
    Find the coordinates of a quote in the PDF.
    Returns the first match found.
    """
    doc = fitz.open(pdf_path)
    # Clean the quote slightly to improve matching chances (remove excess whitespace)
    clean_quote = " ".join(quote.split())
    
    # Search strategy:
    # 1. Try exact match across all pages
    # 2. If not found, try fuzzy match or shorter substrings (simplified here to exact match)
    
    best_rect = None
    page_num = 1
    
    for i, page in enumerate(doc):
        # search_for returns a list of Quad objects (or Rects in older versions, but usually Quads now)
        # We'll use quads to be precise, but for the frontend we need a bounding box (rect)
        matches = page.search_for(clean_quote)
        
        if matches:
            # matches is a list of Rects. If the text spans multiple lines, 
            # we get multiple Rects. We return all of them.
            rects = [[r.x0, r.y0, r.x1, r.y1] for r in matches]
            return {
                "page": i + 1,
                "rects": rects
            }
            
    return None

@router.post("/api/analyze_pdf", response_model=AnalysisResponse)
async def analyze_pdf(
    file: UploadFile = File(...), 
    agents: str = Form("[]"), 
    model: str = Form("gemini-2.5-flash"),
    use_local_model: bool = Form(False),
    scoring_model: str = Form("finetuned")
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
        
    try:
        # 1. Extract text
        text = extract_text_from_pdf(tmp_path)
        
        # 2. Run analysis (LangGraph)
        active_agents = json.loads(agents)
        print(f"Analyzing with model: {model}, Local: {use_local_model}, Scoring Model: {scoring_model}")
        
        api_key = ""
        if not use_local_model:
            if "gemini" in model.lower() and "/" not in model:
                api_key = os.environ.get("GOOGLE_API_KEY")
            else:
                api_key = os.environ.get("OPENROUTER_API_KEY")
            
        review_result = await review_paper(
            text, 
            api_key, 
            active_agents, 
            model=model,
            use_local_model=use_local_model,
            scoring_model=scoring_model
        )

        # 3. Map quotes to coordinates
        annotations = []
        
        # Get page sizes for frontend scaling
        doc = fitz.open(tmp_path)
        page_sizes = []
        for i, page in enumerate(doc):
            page_sizes.append({
                "page": i + 1,
                "width": page.rect.width,
                "height": page.rect.height
            })
            
        for issue in review_result["issues"]:
            # Try to find the quote in the PDF
            coords = find_quote_coordinates(tmp_path, issue["quote"], issue.get("location_hint"))
            
            if coords:
                annotations.append({
                    "id": issue["problem_id"],
                    "category": issue["agent_id"].replace("_agent", "").title(),
                    "agent": issue["agent_id"],
                    "quote": issue["quote"],
                    "comment": f"{issue['suggestion']} (Severity: {issue['severity']})",
                    "page": coords["page"],
                    "rects": coords["rects"],
                    "severity": issue["severity"]
                })
            else:
                # If we can't find the coordinate, we might still want to show it in a sidebar list
                # But for now, we only return mapped annotations as per requirement for highlights
                pass
                
        return {
            "final_score": review_result["final_score"],
            "scoring_summary": review_result.get("scoring_summary"),
            "annotations": annotations,
            "page_sizes": page_sizes,
            "statistics": review_result["statistics"]
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

