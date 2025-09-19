from __future__ import annotations

import io
import os
from tempfile import NamedTemporaryFile
from typing import List, Dict, Any

import fitz  # PyMuPDF
from fastapi import APIRouter, UploadFile, File, HTTPException
from .agents import run_agents_to_annotations


router = APIRouter(prefix="/api", tags=["api"])


@router.post("/analyze_pdf")
async def analyze_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file upload")

    with NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        # open to validate and compute page sizes (in PDF points)
        doc = fitz.open(stream=io.BytesIO(data), filetype="pdf")
        page_sizes = [
            {"page": i + 1, "width": float(p.rect.width), "height": float(p.rect.height)}
            for i, p in enumerate(doc)
        ]
        doc.close()
        annotations = run_agents_to_annotations(tmp_path)
        # Convert to 1-based pages for the PDF.js viewer in frontend_style
        for a in annotations:
            a["page"] = int(a.get("page", 0)) + 1
        return {"annotations": annotations, "page_sizes": page_sizes}
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# Removed agent management endpoints
