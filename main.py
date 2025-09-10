import os
import json
import argparse
import re
from typing import TypedDict, List, Dict, Any
import fitz  # PyMuPDF
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# --- Environment Setup ---
load_dotenv()
# Ensure GOOGLE_API_KEY is present
if not os.getenv("GOOGLE_API_KEY"):
    raise EnvironmentError("GOOGLE_API_KEY environment variable not set in .env file.")

# --- Structured Feedback Definition ---
class FeedbackItem(TypedDict):
    """A structured format for a single piece of feedback."""
    agent: str
    quote: str
    comment: str

# --- Agent State Definition ---
class AgentState(TypedDict):
    """Represents the state of our multi-agent crew."""
    original_text: str
    input_pdf_path: str
    output_pdf_path: str
    feedback: List[FeedbackItem]
    agent_log: List[str]

# --- Helper function for robust JSON extraction ---
def extract_json_from_string(text: str) -> list:
    """Extracts a JSON list from a string, even with markdown fences."""
    # Use regex to find content between ```json and ``` or just a plain list
    match = re.search(r"```json\s*([\s\S]*?)\s*```|(\[[\s\S]*\])", text)
    if match:
        # If the first group (```json) is found, use it. Otherwise, use the second group ([]).
        json_str = match.group(1) if match.group(1) else match.group(2)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print(f"Warning: Failed to decode JSON after extraction: {json_str}")
            return []
    return []

# --- AI Agent Definitions ---
def create_agent(llm, system_prompt: str, agent_name: str):
    """Factory function to create a new agent that expects structured JSON output."""
    def agent_function(state: AgentState) -> AgentState:
        print(f">>> EXECUTING {agent_name.upper()} AGENT <<<")
        
        prompt = f"{system_prompt}\n\nHere is the academic text to review:\n'''\n{state['original_text']}\n'''"
        
        response_message = llm.invoke(prompt)
        response_content = response_message.content
        
        response_json = extract_json_from_string(response_content)
        
        current_feedback = state.get("feedback", [])
        if isinstance(response_json, list):
            for item in response_json:
                current_feedback.append(FeedbackItem(
                    agent=agent_name,
                    quote=item.get("quote", "").strip(),
                    comment=item.get("comment", "").strip()
                ))
        else:
            print(f"Warning: {agent_name} did not return a valid list. Got: {response_json}")

        current_log = state.get("agent_log", [])
        current_log.append(agent_name)

        return {**state, "feedback": current_feedback, "agent_log": current_log}
    return agent_function

# --- NEW: Coordinator Agent ---
def coordinator_agent(state: AgentState) -> AgentState:
    """
    Analyzes all collected feedback, removes duplicates, and filters for relevance.
    """
    print(">>> EXECUTING COORDINATOR AGENT <<<")
    
    # If there's no feedback, no need to run the LLM
    if not state["feedback"]:
        print("No feedback to coordinate. Skipping.")
        return state

    feedback_json_string = json.dumps(state["feedback"], indent=2)
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    
    prompt = f"""
You are an expert Senior Editor responsible for coordinating feedback for a scientific paper.
You have received a list of comments from several specialist agents. Your task is to:
1.  Identify and remove duplicate or redundant suggestions. For example, if a grammar and style agent flag the same sentence for different reasons, consolidate them.
2.  Filter out low-impact or purely subjective comments, retaining only the most critical and actionable feedback.
3.  Ensure the final feedback is clear, concise, and helpful for the author.
4.  Return a cleaned, final list of feedback items in the exact same JSON format as the input.

Here is the raw feedback from the agents:
{feedback_json_string}

Now, provide the final, coordinated list of feedback. Your response MUST be only the JSON list of objects, with no other text or explanation.
"""
    
    response_message = llm.invoke(prompt)
    response_content = response_message.content
    
    final_feedback_json = extract_json_from_string(response_content)
    
    print(f"Coordinator reduced feedback from {len(state['feedback'])} to {len(final_feedback_json)} items.")

    return {**state, "feedback": final_feedback_json}

# --- PDF Processing Node ---
def process_pdf_and_save(state: AgentState) -> AgentState:
    """
    Takes the final, coordinated feedback, highlights quotes in the PDF, adds a summary page,
    and saves the new file.
    """
    print(">>> EXECUTING PDF PROCESSOR <<<")
    input_path = state["input_pdf_path"]
    output_path = state["output_pdf_path"]
    feedback = state.get("feedback", [])
    doc = None # Define doc here to ensure it exists for the finally block

    if not feedback:
        print("No final feedback to add to the PDF. Saving a copy of the original.")
        try:
            original_doc = fitz.open(input_path)
            original_doc.save(output_path)
            original_doc.close()
        except Exception as e:
            print(f"Error saving copy of PDF: {e}")
        return state

    try:
        doc = fitz.open(input_path)

        # 1. Highlight quotes and add hoverable comments
        for item in feedback:
            if item.get("quote"):
                for page in doc:
                    text_instances = page.search_for(item["quote"].strip())
                    for inst in text_instances:
                        # Create the highlight annotation and attach popup
                        highlight = page.add_highlight_annot(inst)
                        highlight.update(info={
                            "content": item['comment'],
                            "title": f"Feedback from {item['agent']}"
                        })

        # 2. Create summary page
        summary_text = "Automated Analysis Feedback Summary\n\nThis summary contains coordinated feedback from multiple AI agents.\n\n"
        for item in feedback:
            summary_text += f"--- Feedback from {item.get('agent', 'Unknown Agent')} ---\n"
            if item.get('quote'):
                summary_text += f"Quoted Text: \"{item['quote']}\"\n"
            summary_text += f"Comment: {item['comment']}\n\n"

        summary_page = doc.new_page(pno=-1, width=doc[0].rect.width, height=doc[0].rect.height)
        rect = fitz.Rect(50, 50, doc[0].rect.width - 50, doc[0].rect.height - 50)
        summary_page.insert_textbox(rect, summary_text, fontsize=10, fontname="helv")

        doc.save(output_path, garbage=4, deflate=True)
        print(f"\nSuccessfully created annotated PDF: {output_path}")

    except Exception as e:
        print(f"An error occurred during PDF processing: {e}")
    finally:
        if doc:
            doc.close()

    return state


# --- Utility: run analysis and return highlight rectangles (for web server) ---
def run_analysis_to_annotations(input_pdf: str) -> List[Dict[str, Any]]:
    """Run the pipeline and return a list of annotations with page & rect coordinates.

    Each item: { page: int, rect: [x0,y0,x1,y1], agent: str, quote: str, comment: str }
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    json_prompt_suffix = (
        """
Your response MUST be a valid JSON list of objects. Each object should have two keys: 'quote' and 'comment'.
- 'quote': The exact, verbatim text snippet from the document that your feedback pertains to.
- 'comment': Your analysis or suggestion regarding that specific quote.
If you find no issues, return an empty list: [].
"""
    )

    proofreader_agent = create_agent(llm, "You are a meticulous proofreader. Identify grammatical errors, spelling mistakes, and typos." + json_prompt_suffix, "Proofreader")
    style_agent = create_agent(llm, "You are a scientific style editor. Assess the text for clarity, conciseness, and academic tone." + json_prompt_suffix, "Stylist")
    coherence_agent = create_agent(llm, "You are a logical coherence analyst. Evaluate the structure and flow of the argument, looking for logical gaps or contradictions." + json_prompt_suffix, "Coherence Analyst")

    workflow = StateGraph(AgentState)
    workflow.add_node("proofreader", proofreader_agent)
    workflow.add_node("stylist", style_agent)
    workflow.add_node("coherence_analyst", coherence_agent)
    workflow.add_node("coordinator", coordinator_agent)

    # Minimal processor to end the graph
    def passthrough(state: AgentState) -> AgentState:
        return state
    workflow.add_node("end_node", passthrough)

    workflow.set_entry_point("proofreader")
    workflow.add_edge("proofreader", "stylist")
    workflow.add_edge("stylist", "coherence_analyst")
    workflow.add_edge("coherence_analyst", "coordinator")
    workflow.add_edge("coordinator", "end_node")
    workflow.add_edge("end_node", END)

    app = workflow.compile()

    try:
        doc = fitz.open(input_pdf)
        full_text = "".join(page.get_text() for page in doc)
        if not full_text.strip():
            return []
        initial_state: AgentState = {
            "original_text": full_text,
            "input_pdf_path": input_pdf,
            "output_pdf_path": "",
            "feedback": [],
            "agent_log": [],
        }
        final_state = app.invoke(initial_state)
        feedback = final_state.get("feedback", []) if isinstance(final_state, dict) else []
        # Map quotes to rects
        annotations: List[Dict[str, Any]] = []
        for page_index, page in enumerate(doc):
            for item in feedback:
                q = (item.get("quote") or "").strip()
                if not q:
                    continue
                for inst in page.search_for(q):
                    annotations.append({
                        "page": page_index,
                        "rect": [inst.x0, inst.y0, inst.x1, inst.y1],
                        "agent": item.get("agent", ""),
                        "quote": q,
                        "comment": item.get("comment", ""),
                    })
        doc.close()
        return annotations
    except Exception:
        try:
            doc.close()
        except Exception:
            pass
        return []


# --- Main Application Logic ---
def run_analysis(input_pdf: str, output_pdf: str):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        # No need for convert_system_message_to_human with modern Gemini models
    )

    json_prompt_suffix = """
Your response MUST be a valid JSON list of objects. Each object should have two keys: 'quote' and 'comment'.
- 'quote': The exact, verbatim text snippet from the document that your feedback pertains to.
- 'comment': Your analysis or suggestion regarding that specific quote.
Example: [{"quote": "the datas shows", "comment": "This should be 'data show' as 'data' is plural."}]
If you find no issues, return an empty list: []. Do not include any text or markdown formatting outside of the JSON list itself.
"""

    proofreader_agent = create_agent(llm, "You are a meticulous proofreader. Identify grammatical errors, spelling mistakes, and typos." + json_prompt_suffix, "Proofreader")
    style_agent = create_agent(llm, "You are a scientific style editor. Assess the text for clarity, conciseness, and academic tone." + json_prompt_suffix, "Stylist")
    coherence_agent = create_agent(llm, "You are a logical coherence analyst. Evaluate the structure and flow of the argument, looking for logical gaps or contradictions." + json_prompt_suffix, "Coherence Analyst")

    # --- UPDATED Graph Workflow ---
    workflow = StateGraph(AgentState)
    
    # Add all nodes to the graph
    workflow.add_node("proofreader", proofreader_agent)
    workflow.add_node("stylist", style_agent)
    workflow.add_node("coherence_analyst", coherence_agent)
    workflow.add_node("coordinator", coordinator_agent) # The new coordinator node
    workflow.add_node("pdf_processor", process_pdf_and_save)

    # Define the sequence of execution
    workflow.set_entry_point("proofreader")
    workflow.add_edge("proofreader", "stylist")
    workflow.add_edge("stylist", "coherence_analyst")
    workflow.add_edge("coherence_analyst", "coordinator") # Run coordinator after all checkers
    workflow.add_edge("coordinator", "pdf_processor")   # Then process the PDF
    workflow.add_edge("pdf_processor", END)

    app = workflow.compile()

    # --- Execution ---
    print("--- Starting PDF Analysis Crew ---")
    try:
        doc = fitz.open(input_pdf)
        full_text = "".join(page.get_text() for page in doc)
        doc.close()
    except Exception as e:
        print(f"Error reading PDF '{input_pdf}': {e}")
        return

    if not full_text.strip():
        print("Error: Could not extract text from the PDF.")
        return

    initial_state = {
        "original_text": full_text,
        "input_pdf_path": input_pdf,
        "output_pdf_path": output_pdf,
        "feedback": [],
        "agent_log": []
    }

    app.invoke(initial_state)

    print("\n--- Analysis Complete ---")

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze an academic PDF with a multi-agent crew.")
    parser.add_argument("input_pdf", help="The path to the input PDF file.")
    parser.add_argument("output_pdf", help="The path to save the annotated output PDF file.")
    args = parser.parse_args()
    
    run_analysis(args.input_pdf, args.output_pdf)