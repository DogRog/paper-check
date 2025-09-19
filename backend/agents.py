from __future__ import annotations

import json
import re
import uuid
from threading import Lock
from typing import Any, Dict, List, TypedDict, Optional

import fitz  # PyMuPDF
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END


class FeedbackItem(TypedDict):
    agent: str
    quote: str
    comment: str
    category: str


class AgentState(TypedDict):
    original_text: str
    input_pdf_path: str
    feedback: List[FeedbackItem]
    agent_log: List[str]


class AgentConfig(TypedDict):
    id: str
    name: str
    category: str  # 'tone' | 'structure' | 'coherence' | 'other'
    prompt: str


_LOCK = Lock()
_AGENTS: Dict[str, AgentConfig] = {}


def _init_defaults() -> None:
    if _AGENTS:
        return
    defaults: List[AgentConfig] = [
        {
            "id": "stylist",
            "name": "Stylist",
            "category": "tone",
            "prompt": (
                "You are a scientific style editor. Identify unclear, verbose, or non-academic tone. "
                "Avoid generic praise; only return specific issues with actionable suggestions."
            ),
        },
        {
            "id": "structure",
            "name": "Structure Reviewer",
            "category": "structure",
            "prompt": (
                "You are a structure reviewer. Assess section/paragraph organization, headings, ordering, and transitions. "
                "Flag structural issues such as misplaced content, weak transitions, and poor paragraph focus."
            ),
        },
        {
            "id": "coherence",
            "name": "Coherence Analyst",
            "category": "coherence",
            "prompt": (
                "You are a logical coherence analyst. Evaluate argument flow, logical gaps, contradictions, and missing links. "
                "Only return concrete issues tied to exact quotes."
            ),
        },
    ]
    with _LOCK:
        for a in defaults:
            _AGENTS[a["id"]] = a


def list_agents() -> List[AgentConfig]:
    _init_defaults()
    with _LOCK:
        return list(_AGENTS.values())


def create_agent_cfg(name: str, category: str, prompt: str) -> AgentConfig:
    _init_defaults()
    aid = uuid.uuid4().hex[:8]
    cfg: AgentConfig = {"id": aid, "name": name, "category": category, "prompt": prompt}
    with _LOCK:
        _AGENTS[aid] = cfg
    return cfg


def update_agent_cfg(agent_id: str, name: Optional[str] = None, category: Optional[str] = None, prompt: Optional[str] = None) -> AgentConfig:
    _init_defaults()
    with _LOCK:
        if agent_id not in _AGENTS:
            raise KeyError("Agent not found")
        cfg = dict(_AGENTS[agent_id])
        if name is not None:
            cfg["name"] = name
        if category is not None:
            cfg["category"] = category
        if prompt is not None:
            cfg["prompt"] = prompt
        _AGENTS[agent_id] = cfg  # type: ignore
        return cfg  # type: ignore


def delete_agent_cfg(agent_id: str) -> None:
    _init_defaults()
    with _LOCK:
        if agent_id in _AGENTS:
            del _AGENTS[agent_id]


def _extract_json(text: str) -> List[Dict[str, str]]:
    m = re.search(r"```json\s*([\s\S]*?)\s*```|(\[[\s\S]*\])", text)
    if not m:
        return []
    s = m.group(1) if m.group(1) else m.group(2)
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return obj
    except Exception:
        return []
    return []


def build_agent_node(llm, system_prompt: str, agent_name: str, category: str):
    def agent_fn(state: AgentState) -> AgentState:
        msg = llm.invoke(
            f"{system_prompt}\n\nConstraints:\n- Only include issues that require improvement.\n- Do NOT include praise-only items like 'well written', 'looks good', 'no changes'.\n- Each item MUST include a verbatim problematic quote from the text.\n\nText to review:\n'''\n{state['original_text']}\n'''"
        )
        content = getattr(msg, "content", str(msg))
        items = _extract_json(content)

        def _trivial(c: str) -> bool:
            c_low = (c or "").strip().lower()
            if not c_low:
                return True
            bad_phrases = [
                "well written",
                "well-written",
                "looks good",
                "no changes",
                "no issue",
                "no issues",
                "nothing to change",
                "clear as is",
                "reads well",
            ]
            return any(p in c_low for p in bad_phrases)

        fb = state.get("feedback", [])
        for it in items:
            q = (it.get("quote") or "").strip()
            c = (it.get("comment") or "").strip()
            if not q or _trivial(c):
                continue
            fb.append(FeedbackItem(agent=agent_name, quote=q, comment=c, category=category))
        log = state.get("agent_log", [])
        log.append(agent_name)
        return {**state, "feedback": fb, "agent_log": log}

    return agent_fn


def coordinator_agent(state: AgentState) -> AgentState:
    if not state.get("feedback"):
        return state
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    prompt = (
        "You consolidate overlapping, redundant, or low-value feedback. Return only the JSON list.\n"
        f"Input: {json.dumps(state['feedback'])}"
    )
    msg = llm.invoke(prompt)
    content = getattr(msg, "content", str(msg))
    final = _extract_json(content)
    # If consolidation loses category/agent, fall back to original list
    try:
        if final and isinstance(final[0], dict) and ("agent" not in final[0] or "category" not in final[0]):
            return state
    except Exception:
        return state
    return {**state, "feedback": final} if final else state


def run_agents_to_annotations(input_pdf_path: str) -> List[Dict[str, Any]]:
    """Run configured agents and map quotes to PDF rectangles."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    json_suffix = (
        """
Your response MUST be a valid JSON list of objects. Each object should have two keys: 'quote' and 'comment'.
- 'quote': The exact, verbatim text snippet from the document that your feedback pertains to.
- 'comment': Your analysis or suggestion regarding that specific quote.
If you find no issues, return an empty list: [].
"""
    )

    # Build graph dynamically from configured agents
    configs = list_agents()
    graph = StateGraph(AgentState)
    node_names: List[str] = []
    for idx, cfg in enumerate(configs):
        node_name = f"agent_{idx}"
        node = build_agent_node(llm, (cfg["prompt"] + json_suffix), cfg["name"], cfg.get("category", "other"))
        graph.add_node(node_name, node)
        node_names.append(node_name)
    graph.add_node("coordinator", coordinator_agent)
    if node_names:
        graph.set_entry_point(node_names[0])
        # chain sequentially then coordinator
        for a, b in zip(node_names, node_names[1:]):
            graph.add_edge(a, b)
        graph.add_edge(node_names[-1], "coordinator")
    else:
        # No agents configured; nothing to do
        graph.set_entry_point("coordinator")
    graph.add_edge("coordinator", END)
    app = graph.compile()

    annotations: List[Dict[str, Any]] = []
    doc = fitz.open(input_pdf_path)
    try:
        full_text = "".join(page.get_text() for page in doc)
        if not full_text.strip():
            return []
        state: AgentState = {
            "original_text": full_text,
            "input_pdf_path": input_pdf_path,
            "feedback": [],
            "agent_log": [],
        }
        final_state = app.invoke(state)
        feedback = final_state.get("feedback", []) if isinstance(final_state, dict) else []
        for page_index, page in enumerate(doc):
            for item in feedback:
                q = (item.get("quote") or "").strip()
                if not q:
                    continue
                for inst in page.search_for(q):
                    annotations.append(
                        {
                            "page": page_index,
                            "rect": [inst.x0, inst.y0, inst.x1, inst.y1],
                            "agent": item.get("agent", ""),
                            "category": item.get("category", "other"),
                            "quote": q,
                            "comment": item.get("comment", ""),
                        }
                    )
        return annotations
    finally:
        doc.close()
