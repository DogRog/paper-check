"""
Academic Paper Review System using LangGraph
Analyzes papers for tone, structure, coherence, and citations
"""

import json
import uuid
import asyncio
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph


def clean_json_text(text: str | BaseMessage) -> str:
    """Clean JSON text by removing markdown code blocks"""
    if hasattr(text, 'content'):
        text = text.content
    text = text.strip()
    
    # Try to find markdown code blocks first
    match = re.search(r"```(?:json)?\s*(.*)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
        
    # If no code blocks, try to find the first [ and last ]
    start = text.find('[')
    end = text.rfind(']')
    
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
        
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
    location_hint: Optional[str] = None  # For tracking overlapping regions
    
    def to_dict(self):
        return {
            "problem_id": self.problem_id,
            "agent_id": self.agent_id,
            "quote": self.quote,
            "suggestion": self.suggestion,
            "severity": self.severity
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

    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        
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
                response = await chain.ainvoke({"text": state["paper_text"]})
                issues_data = json.loads(clean_json_text(response))
                print(f"Tone Agent Output: {json.dumps(issues_data, indent=2)}")
                
                tone_issues = []
                for issue in issues_data:
                    tone_issues.append(Issue(
                        problem_id=str(uuid.uuid4()),
                        agent_id="tone_agent",
                        quote=issue["quote"],
                        suggestion=issue["suggestion"],
                        severity=SeverityLevel(issue["severity"]),
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
                print(f"Structure Agent Output: {json.dumps(issues_data, indent=2)}")
                
                structure_issues = []
                for issue in issues_data:
                    structure_issues.append(Issue(
                        problem_id=str(uuid.uuid4()),
                        agent_id="structure_agent",
                        quote=issue["quote"],
                        suggestion=issue["suggestion"],
                        severity=SeverityLevel(issue["severity"]),
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
                print(f"Coherence Agent Output: {json.dumps(issues_data, indent=2)}")
                
                coherence_issues = []
                for issue in issues_data:
                    coherence_issues.append(Issue(
                        problem_id=str(uuid.uuid4()),
                        agent_id="coherence_agent",
                        quote=issue["quote"],
                        suggestion=issue["suggestion"],
                        severity=SeverityLevel(issue["severity"]),
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
                print(f"Citation Agent Output: {json.dumps(issues_data, indent=2)}")
                
                citation_issues = []
                for issue in issues_data:
                    citation_issues.append(Issue(
                        problem_id=str(uuid.uuid4()),
                        agent_id="citation_agent",
                        quote=issue["quote"],
                        suggestion=issue["suggestion"],
                        severity=SeverityLevel(issue["severity"]),
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
        
        def process(state: GraphState) -> dict:
            # Collect all issues
            all_issues = {
                "tone_issues": [issue.to_dict() for issue in state.get("tone_issues", [])],
                "structure_issues": [issue.to_dict() for issue in state.get("structure_issues", [])],
                "coherence_issues": [issue.to_dict() for issue in state.get("coherence_issues", [])],
                "citation_issues": [issue.to_dict() for issue in state.get("citation_issues", [])]
            }
            
            try:
                response = chain.invoke({"issues_json": json.dumps(all_issues, indent=2)})
                coordinated_data = json.loads(clean_json_text(response))
                print(f"Coordinator Agent Output: {json.dumps(coordinated_data, indent=2)}")
                
                coordinated_issues = []
                for issue in coordinated_data:
                    coordinated_issues.append(Issue(
                        problem_id=issue.get("problem_id", str(uuid.uuid4())),
                        agent_id=issue["agent_id"],
                        quote=issue["quote"],
                        suggestion=issue["suggestion"],
                        severity=SeverityLevel(issue["severity"]),
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
        """Agent that provides final score for the paper"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a senior academic reviewer providing a final score for the paper.
            
            Based on the coordinated issues list, evaluate the paper's overall quality:
            
            Scoring criteria (1-10 scale):
            - 9-10: Exceptional paper, publication-ready with minimal revisions
            - 7-8: Good paper, needs minor revisions
            - 5-6: Adequate paper, needs moderate revisions
            - 3-4: Poor paper, needs major revisions
            - 1-2: Very poor paper, needs complete rewrite
            
            Consider:
            - Number and severity of issues
            - Overall readability and structure
            - Academic rigor and coherence
            - Citation quality
            
            Output as JSON:
            {{
                "score": <number 1-10>,
                "summary": "brief explanation of score"
            }}"""),
            ("human", "Issues found:\n{issues_summary}\n\nOriginal paper excerpt:\n{paper_excerpt}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        def process(state: GraphState) -> dict:
            # Prepare issues summary
            issues_summary = []
            severity_counts = {"high": 0, "medium": 0, "low": 0}
            
            for issue in state.get("coordinated_issues", []):
                issues_summary.append(f"- [{issue.severity}] {issue.agent_id}: {issue.quote[:50]}...")
                severity_counts[issue.severity] += 1
            
            issues_text = "\n".join(issues_summary) if issues_summary else "No major issues found"
            paper_excerpt = state["paper_text"][:1000]  # First 1000 chars as context
            
            try:
                response = chain.invoke({
                    "issues_summary": issues_text,
                    "paper_excerpt": paper_excerpt
                })
                score_data = json.loads(clean_json_text(response))
                print(f"Scoring Agent Output: {json.dumps(score_data, indent=2)}")
                
                # Add summary to messages
                new_messages = state["messages"] + [
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
                base_score = 10
                base_score -= severity_counts["high"] * 2
                base_score -= severity_counts["medium"] * 1
                base_score -= severity_counts["low"] * 0.5
                return {
                    "final_score": max(1, min(10, base_score)),
                    "scoring_summary": "Score calculated based on issue severity counts due to processing error."
                }
        
        return process



def create_review_graph(llm: ChatGoogleGenerativeAI, active_agents: List[str] = None) -> StateGraph:
    """Create the LangGraph workflow for paper review"""
    
    # Initialize agents
    agents = PaperReviewAgents(llm)
    
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



async def review_paper(paper_text: str, api_key: str, active_agents: List[str] = None, model: str = "gemini-2.5-flash") -> Dict:
    """
    Main function to review a paper
    
    Args:
        paper_text: Text extracted from paper (e.g., using docling)
        api_key: API key (Google or OpenRouter)
        active_agents: List of agents to run (tone, structure, coherence, citation)
        model: Model name to use
    
    Returns:
        Dictionary with review results
    """
    # Initialize LLM
    if "gemini" in model.lower() and "/" not in model:
        llm = ChatGoogleGenerativeAI(
            temperature=0.0,
            model=model,
            api_key=api_key
        )
    else:
        # Assume OpenRouter for other models
        llm = ChatOpenAI(
            temperature=0.0,
            model=model,
            openai_api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
    
    # Create the graph
    app = create_review_graph(llm, active_agents)
    
    # Initialize state
    initial_state = {
        "paper_text": paper_text,
        "tone_issues": [],
        "structure_issues": [],
        "scoring_summary": None,
        "messages": []
    }
    
    # Run the graph
    result = await app.ainvoke(initial_state)
    
    # Format output
    output = {
        "issues": [issue.to_dict() for issue in result["coordinated_issues"]],
        "final_score": result["final_score"],
        "scoring_summary": result.get("scoring_summary"),
        "statistics": {
            "total_issues": len(result["coordinated_issues"]),
            "by_severity": {
                "high": sum(1 for i in result["coordinated_issues"] if i.severity == SeverityLevel.HIGH),
                "medium": sum(1 for i in result["coordinated_issues"] if i.severity == SeverityLevel.MEDIUM),
                "low": sum(1 for i in result["coordinated_issues"] if i.severity == SeverityLevel.LOW)
            },
            "by_agent": {
                "tone": sum(1 for i in result["coordinated_issues"] if i.agent_id == "tone_agent"),
                "structure": sum(1 for i in result["coordinated_issues"] if i.agent_id == "structure_agent"),
                "coherence": sum(1 for i in result["coordinated_issues"] if i.agent_id == "coherence_agent"),
                "citation": sum(1 for i in result["coordinated_issues"] if i.agent_id == "citation_agent")
            }
        }
    }
    
    return output
