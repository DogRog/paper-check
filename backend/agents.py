import json
import uuid
import asyncio
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, TypedDict

# Added SystemMessage to imports
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# Assuming these exist in your backend
from backend.llm import get_llm, score_paper


def clean_json_text(text: str | BaseMessage) -> str:
    """
    Clean JSON text by removing markdown code blocks and finding the correct 
    outermost brackets (either [] for arrays or {} for objects).
    """
    if hasattr(text, 'content'):
        text = text.content
    text = text.strip()
    
    # 1. Try to find markdown code blocks first
    match = re.search(r"```(?:json)?\s*(.*)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
        
    # 2. Determine if we are looking for an Array or an Object
    # We look for the first occurrence of [ or {
    first_sq = text.find('[')
    first_curly = text.find('{')
    
    # If no brackets found, return original (will likely fail parse)
    if first_sq == -1 and first_curly == -1:
        return text

    # Logic to extract the correct JSON structure
    if first_sq != -1 and (first_curly == -1 or first_sq < first_curly):
        # It's an array
        end = text.rfind(']')
        if end != -1:
            return text[first_sq:end+1]
    else:
        # It's an object
        end = text.rfind('}')
        if end != -1:
            return text[first_curly:end+1]
            
    return text


class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Issue:
    """Represents an issue found by an agent"""
    problem_id: str
    agent_id: str
    quote: str
    suggestion: str
    severity: SeverityLevel
    location_hint: Optional[str] = None
    
    def to_dict(self):
        return {
            "problem_id": self.problem_id,
            "agent_id": self.agent_id,
            "quote": self.quote,
            "suggestion": self.suggestion,
            "severity": self.severity,
            "location_hint": self.location_hint
        }


class GraphState(TypedDict):
    """State passed between nodes in the graph"""
    paper_text: str
    tone_issues: List[Issue]
    structure_issues: List[Issue]
    coherence_issues: List[Issue]
    citation_issues: List[Issue]
    coordinated_issues: List[Issue]
    final_score: Optional[float]
    scoring_summary: Optional[str]
    messages: List[BaseMessage]


class PaperReviewAgents:
    """Collection of agents for paper review"""

    def __init__(self, llm: BaseChatModel, scoring_model: str = "finetuned"):
        self.llm = llm
        self.scoring_model = scoring_model
        
    def create_tone_agent(self):
        """Agent that checks for non-academic tone"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert academic tone reviewer. Analyze the paper text for:
            - Informal language or colloquialisms
            - Inappropriate use of personal pronouns in formal sections
            - Emotional or biased language
            - Overly casual expressions
            - Marketing or promotional language
            
            For each issue found, provide:
            1. An exact quote of the problematic text (30-150 characters)
            2. Severity: low (minor style issue), medium (noticeable problem), high (major tone violation)
            3. A specific suggestion for improvement
            
            Output as JSON array with format:
            [{{
                "quote": "exact problematic text",
                "severity": "low/medium/high",
                "suggestion": "specific improvement suggestion",
                "location_hint": "section or paragraph description"
            }}]
            
            If no issues found, return empty array []."""),
            ("human", "{text}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        async def process(state: GraphState) -> dict:
            try:
                # Note: We do NOT format {text} here with f-string to allow LangChain to handle escaping
                response = await chain.ainvoke({"text": state["paper_text"]})
                issues_data = json.loads(clean_json_text(response))
                print(f"Tone Agent Output: {len(issues_data)} issues found")
                
                tone_issues = []
                for issue in issues_data:
                    tone_issues.append(Issue(
                        problem_id=str(uuid.uuid4()),
                        agent_id="tone_agent",
                        quote=issue.get("quote", ""),
                        suggestion=issue.get("suggestion", ""),
                        severity=SeverityLevel(issue.get("severity", "low")),
                        location_hint=issue.get("location_hint")
                    ))
                
                return {"tone_issues": tone_issues}
            except Exception as e:
                print(f"Tone agent error: {e}")
                return {"tone_issues": []}
        
        return process
    
    def create_structure_agent(self):
        """Agent that checks paper structure"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in academic paper structure. Analyze the paper for:
            - Missing or poorly organized sections (Abstract, Introduction, Methods, Results, Discussion, Conclusion)
            - Imbalanced section lengths
            - Poor paragraph structure or transitions
            - Missing or inadequate abstract components
            - Inappropriate content placement
            
            For each issue found, provide:
            1. An exact quote showing the problem (30-150 characters) or description if structural
            2. Severity: low (minor organization), medium (notable structure issue), high (major structural flaw)
            3. A specific suggestion for improvement
            
            Output as JSON array with format:
            [{{
                "quote": "problematic text or section description",
                "severity": "low/medium/high",
                "suggestion": "specific improvement suggestion",
                "location_hint": "section or location"
            }}]
            
            If no issues found, return empty array []."""),
            ("human", "{text}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        async def process(state: GraphState) -> dict:
            try:
                response = await chain.ainvoke({"text": state["paper_text"]})
                issues_data = json.loads(clean_json_text(response))
                print(f"Structure Agent Output: {len(issues_data)} issues found")
                
                structure_issues = []
                for issue in issues_data:
                    structure_issues.append(Issue(
                        problem_id=str(uuid.uuid4()),
                        agent_id="structure_agent",
                        quote=issue.get("quote", ""),
                        suggestion=issue.get("suggestion", ""),
                        severity=SeverityLevel(issue.get("severity", "low")),
                        location_hint=issue.get("location_hint")
                    ))
                
                return {"structure_issues": structure_issues}
            except Exception as e:
                print(f"Structure agent error: {e}")
                return {"structure_issues": []}
        
        return process
    
    def create_coherence_agent(self):
        """Agent that evaluates argument flow and logical consistency"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in academic argumentation and logic. Analyze the paper for:
            - Logical gaps or jumps in reasoning
            - Contradictions between different sections
            - Missing links between claims and evidence
            - Unsupported conclusions
            - Circular reasoning
            - Unclear cause-effect relationships
            
            For each issue found, provide:
            1. An exact quote showing the problem (30-150 characters)
            2. Severity: low (minor flow issue), medium (notable gap), high (major logical flaw)
            3. A specific suggestion for improvement
            
            Output as JSON array with format:
            [{{
                "quote": "problematic text",
                "severity": "low/medium/high",
                "suggestion": "specific improvement suggestion",
                "location_hint": "section or context"
            }}]
            
            If no issues found, return empty array []."""),
            ("human", "{text}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        async def process(state: GraphState) -> GraphState:
            try:
                response = await chain.ainvoke({"text": state["paper_text"]})
                issues_data = json.loads(clean_json_text(response))
                print(f"Coherence Agent Output: {len(issues_data)} issues found")
                
                coherence_issues = []
                for issue in issues_data:
                    coherence_issues.append(Issue(
                        problem_id=str(uuid.uuid4()),
                        agent_id="coherence_agent",
                        quote=issue.get("quote", ""),
                        suggestion=issue.get("suggestion", ""),
                        severity=SeverityLevel(issue.get("severity", "low")),
                        location_hint=issue.get("location_hint")
                    ))
                
                return {"coherence_issues": coherence_issues}
            except Exception as e:
                print(f"Coherence agent error: {e}")
                return {"coherence_issues": []}
        
        return process

    def create_citation_agent(self):
        """Agent that checks citation placement and relevance"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in academic citations. Analyze the paper for:
            - Missing citations for claims that need support
            - Incorrectly placed citations
            - Over-citation or under-citation
            - Irrelevant or inappropriate citations
            - Self-citation issues
            - Citation format consistency
            
            For each issue found, provide:
            1. An exact quote showing the problem (30-150 characters)
            2. Severity: low (minor citation issue), medium (notable problem), high (major citation flaw)
            3. A specific suggestion for improvement
            
            Output as JSON array with format:
            [{{
                "quote": "problematic text",
                "severity": "low/medium/high",
                "suggestion": "specific improvement suggestion",
                "location_hint": "section or context"
            }}]
            
            If no issues found, return empty array []."""),
            ("human", "{text}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        async def process(state: GraphState) -> GraphState:
            try:
                response = await chain.ainvoke({"text": state["paper_text"]})
                issues_data = json.loads(clean_json_text(response))
                print(f"Citation Agent Output: {len(issues_data)} issues found")
                
                citation_issues = []
                for issue in issues_data:
                    citation_issues.append(Issue(
                        problem_id=str(uuid.uuid4()),
                        agent_id="citation_agent",
                        quote=issue.get("quote", ""),
                        suggestion=issue.get("suggestion", ""),
                        severity=SeverityLevel(issue.get("severity", "low")),
                        location_hint=issue.get("location_hint")
                    ))
                
                return {"citation_issues": citation_issues}
            except Exception as e:
                print(f"Citation agent error: {e}")
                return {"citation_issues": []}
        
        return process
    
    def create_coordinator_agent(self):
        """Coordinator agent that reviews and deduplicates issues"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a senior academic reviewer coordinating feedback from multiple agents.
            
            Your tasks:
            1. Review all issues from different agents
            2. Identify overlapping issues (same or very similar quotes/regions)
            3. For overlaps, determine which agent's perspective is most relevant
            4. Validate severity scores and adjust if needed
            5. Filter out false positives or minor issues not worth reporting
            
            For the final issue list, ensure:
            - No duplicate regions/quotes
            - Appropriate severity levels
            - Clear, actionable suggestions
            
            Input format will be a JSON with all agent issues.
            
            Output as JSON array with the same format as input, but coordinated and filtered.
            Include only issues that should be in the final report."""),
            ("human", "{issues_json}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        # Converted to async to avoid blocking
        async def process(state: GraphState) -> dict:
            # Collect all issues
            all_issues = {
                "tone_issues": [issue.to_dict() for issue in state.get("tone_issues", [])],
                "structure_issues": [issue.to_dict() for issue in state.get("structure_issues", [])],
                "coherence_issues": [issue.to_dict() for issue in state.get("coherence_issues", [])],
                "citation_issues": [issue.to_dict() for issue in state.get("citation_issues", [])]
            }
            
            try:
                response = await chain.ainvoke({"issues_json": json.dumps(all_issues, indent=2)})
                coordinated_data = json.loads(clean_json_text(response))
                print(f"Coordinator Agent Output: {len(coordinated_data)} final issues")
                
                coordinated_issues = []
                for issue in coordinated_data:
                    coordinated_issues.append(Issue(
                        problem_id=issue.get("problem_id", str(uuid.uuid4())),
                        agent_id=issue.get("agent_id", "coordinator"),
                        quote=issue.get("quote", ""),
                        suggestion=issue.get("suggestion", ""),
                        severity=SeverityLevel(issue.get("severity", "low")),
                        location_hint=issue.get("location_hint")
                    ))
                
                return {"coordinated_issues": coordinated_issues}
            except Exception as e:
                print(f"Coordinator agent error: {e}")
                # Fallback: combine all issues without coordination
                return {"coordinated_issues": (
                    state.get("tone_issues", []) +
                    state.get("structure_issues", []) +
                    state.get("coherence_issues", []) +
                    state.get("citation_issues", [])
                )}
        
        return process
    
    def create_scoring_agent(self):
        """Agent that provides final score for the paper using fine-tuned Qwen model"""
        
        async def process(state: GraphState) -> dict:
            # Prepare issues summary
            issues_summary = []
            severity_counts = {"high": 0, "medium": 0, "low": 0}
            
            for issue in state.get("coordinated_issues", []):
                # Handle case where issue might be dict or object (safety check)
                if isinstance(issue, dict):
                    sev = issue.get('severity', 'low')
                    agent = issue.get('agent_id', 'unknown')
                    quote = issue.get('quote', '')
                else:
                    sev = issue.severity
                    agent = issue.agent_id
                    quote = issue.quote
                    
                issues_summary.append(f"- [{sev}] {agent}: {quote[:50]}...")
                if sev in severity_counts:
                    severity_counts[sev] += 1
            
            issues_text = "\n".join(issues_summary) if issues_summary else "No major issues found"
            paper_text = state["paper_text"]
            
            try:
                score_data = {}
                
                # Check scoring model strategy
                if self.scoring_model == "finetuned":
                    # Run scoring model in a separate thread to avoid blocking
                    print("Running finetuned scoring model...")
                    score_data = await asyncio.to_thread(
                        score_paper, 
                        paper_text, 
                        issues_text,
                        True # use_finetuned
                    )
                elif self.scoring_model == "base":
                    # Run base local model
                    print("Running base local scoring model...")
                    score_data = await asyncio.to_thread(
                        score_paper, 
                        paper_text, 
                        issues_text,
                        False # use_finetuned=False -> uses base model
                    )
                else:
                    # Use API model (Gemini or OpenRouter)
                    print("Running API scoring model...")
                    
                    # NOTE: We construct the content string manually to avoid f-string
                    # interpretation of LaTeX braces inside paper_text
                    
                    system_instr = "You are an expert academic paper reviewer. Return JSON only."
                    
                    user_content_parts = [
                        "Based on the following issues found in the paper:\n",
                        issues_text,
                        "\n\nAnd the paper text (full text provided):\n",
                        "--- BEGIN PAPER ---\n",
                        paper_text,
                        "\n--- END PAPER ---\n",
                        "\nProvide a final score out of 10 and a brief summary justification.",
                        "\nThe score should reflect the severity and number of issues found.",
                        "\nReturn ONLY a JSON object with the following format:",
                        "\n{",
                        '\n  "score": 1-10,',
                        '\n  "summary": "brief justification text"',
                        "\n}"
                    ]
                    
                    user_content = "".join(user_content_parts)
                    
                    messages = [
                        SystemMessage(content=system_instr),
                        HumanMessage(content=user_content)
                    ]
                    
                    response = await self.llm.ainvoke(messages)
                    content = clean_json_text(response)
                    score_data = json.loads(content)
                    
                    # Ensure required fields and types
                    if "score" not in score_data:
                        score_data["score"] = 5.0
                    else:
                        score_data["score"] = float(score_data["score"])
                        
                    if "summary" not in score_data:
                        score_data["summary"] = "No summary provided."

                print(f"Scoring Agent Output: {score_data.get('score')} - {score_data.get('summary')[:50]}...")
                
                # Add summary to messages
                new_messages = state.get("messages", []) + [
                    HumanMessage(content=f"Final score: {score_data['score']}/10 - {score_data['summary']}")
                ]
                
                return {
                    "final_score": score_data["score"],
                    "scoring_summary": score_data["summary"],
                    "messages": new_messages
                }
            except Exception as e:
                print(f"Scoring agent error: {e}")
                # Fallback scoring based on severity counts
                base_score = 10.0
                base_score -= severity_counts["high"] * 2.0
                base_score -= severity_counts["medium"] * 1.0
                base_score -= severity_counts["low"] * 0.5
                
                final_calc_score = max(1.0, min(10.0, base_score))
                
                return {
                    "final_score": final_calc_score,
                    "scoring_summary": "Score calculated based on issue severity counts due to processing error."
                }
        
        return process


def create_review_graph(llm: BaseChatModel, active_agents: List[str] = None, scoring_model: str = "finetuned") -> StateGraph:
    """Create the LangGraph workflow for paper review"""
    
    # Initialize agents
    agents = PaperReviewAgents(llm, scoring_model)
    
    # Create the graph
    workflow = StateGraph(GraphState)
    
    # Define parallel analysis node
    async def parallel_analysis(state: GraphState):
        tasks = []
        agent_map = {
            "tone": agents.create_tone_agent(),
            "structure": agents.create_structure_agent(),
            "coherence": agents.create_coherence_agent(),
            "citation": agents.create_citation_agent()
        }
        
        # Determine which agents to run
        agents_to_run = active_agents if active_agents else agent_map.keys()
        
        for name in agents_to_run:
            if name in agent_map:
                tasks.append(agent_map[name](state))
        
        if not tasks:
            return {}
            
        results = await asyncio.gather(*tasks)
        
        # Merge results
        new_state = {}
        for res in results:
            new_state.update(res)
        return new_state

    # Add nodes
    workflow.add_node("parallel_analysis", parallel_analysis)
    workflow.add_node("coordinator", agents.create_coordinator_agent())
    workflow.add_node("scorer", agents.create_scoring_agent())
    
    # Define the flow
    workflow.set_entry_point("parallel_analysis")
    workflow.add_edge("parallel_analysis", "coordinator")
    workflow.add_edge("coordinator", "scorer")
    workflow.add_edge("scorer", END)
    
    return workflow.compile()


async def review_paper(paper_text: str, api_key: str, active_agents: List[str] = None, model: str = "gemini-2.0-flash", use_local_model: bool = False, scoring_model: str = "finetuned") -> Dict:
    """
    Main function to review a paper
    
    Args:
        paper_text: Text extracted from paper
        api_key: API key
        active_agents: List of agents to run
        model: Model name to use
        use_local_model: Whether to use local Qwen model for agents
        scoring_model: Which model to use for scoring ("finetuned", "base", "api")
    """
    # Initialize LLM
    llm = get_llm(model=model, api_key=api_key, use_local=use_local_model)
    
    # Create the graph
    app = create_review_graph(llm, active_agents, scoring_model)
    
    # Initialize state
    initial_state = {
        "paper_text": paper_text,
        "tone_issues": [],
        "structure_issues": [],
        "coherence_issues": [],
        "citation_issues": [],
        "scoring_summary": None,
        "messages": []
    }
    
    # Run the graph
    result = await app.ainvoke(initial_state)
    
    # Format output
    output = {
        "issues": [issue.to_dict() for issue in result.get("coordinated_issues", [])],
        "final_score": result.get("final_score"),
        "scoring_summary": result.get("scoring_summary"),
        "statistics": {
            "total_issues": len(result.get("coordinated_issues", [])),
            "by_severity": {
                "high": sum(1 for i in result.get("coordinated_issues", []) if i.severity == SeverityLevel.HIGH),
                "medium": sum(1 for i in result.get("coordinated_issues", []) if i.severity == SeverityLevel.MEDIUM),
                "low": sum(1 for i in result.get("coordinated_issues", []) if i.severity == SeverityLevel.LOW)
            },
            "by_agent": {
                "tone": sum(1 for i in result.get("coordinated_issues", []) if i.agent_id == "tone_agent"),
                "structure": sum(1 for i in result.get("coordinated_issues", []) if i.agent_id == "structure_agent"),
                "coherence": sum(1 for i in result.get("coordinated_issues", []) if i.agent_id == "coherence_agent"),
                "citation": sum(1 for i in result.get("coordinated_issues", []) if i.agent_id == "citation_agent")
            }
        }
    }
    
    return output