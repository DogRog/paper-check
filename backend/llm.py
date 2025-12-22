import json
import re
import os
from typing import Any, Dict, Optional
from openai import OpenAI

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

def get_llm(model: str, api_key: str = None, temperature: float = 0.0, use_local: bool = False):
    """
    Factory function to get the appropriate LLM instance.
    """
    if use_local:
        # Use OpenRouter Qwen 3 14B as "local" model replacement
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            model="qwen/qwen3-14b",
            temperature=temperature
        )
    
    if "gemini" in model.lower() and "/" not in model:
        return ChatGoogleGenerativeAI(
            temperature=temperature,
            model=model,
            api_key=api_key or os.environ.get("GOOGLE_API_KEY")
        )
    else:
        # Assume OpenRouter for other models
        return ChatOpenAI(
            temperature=temperature,
            model=model,
            openai_api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )


def clean_json_text(text: str) -> str:
    """Clean JSON text by removing markdown code blocks"""
    text = text.strip()
    
    # Try to find markdown code blocks first
    match = re.search(r"```(?:json)?\s*(.*)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
        
    # If no code blocks, try to find the first { and last }
    start = text.find('{')
    end = text.rfind('}')
    
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
        
    return text


def extract_score_and_summary(text: str) -> Dict[str, Any]:
    """Fallback extraction if JSON parsing fails"""
    data = {}
    
    # Look for score
    score_match = re.search(r"score[\"']?\s*[:=]\s*[\"']?(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if score_match:
        try:
            data["score"] = float(score_match.group(1))
        except ValueError:
            pass
            
    # Look for summary
    summary_match = re.search(r"summary[\"']?\s*[:=]\s*[\"']?(.*)", text, re.IGNORECASE | re.DOTALL)
    if summary_match:
        summary = summary_match.group(1).strip()
        # Clean up trailing quotes or braces if they were captured
        if summary.endswith('"') or summary.endswith('}'):
            summary = summary[:-1].strip()
        data["summary"] = summary
        
    return data


def score_paper(paper_text: str, issues_text: str, use_finetuned: bool = True, llm: Optional[Any] = None) -> Dict[str, Any]:
    """
    Score the paper using either the finetuned model (HF Endpoint), base model (OpenRouter), or provided LLM.
    """
    
    prompt = f"""You are an expert academic paper reviewer.
    
    Based on the following issues found in the paper:
    {issues_text}
    
    And the paper text:
    {paper_text}
    
    Provide a final score out of 10 and a brief summary justification.
    The score should reflect the severity and number of issues found.
    
    Return ONLY a JSON object with the following format:
    {{
        "score": 1-10,
        "summary": "brief justification text"
    }}
    """

    try:
        content = ""
        
        if use_finetuned:
            # Use HF Endpoint via OpenAI client
            client = OpenAI(
                base_url=os.environ.get("INSTANCE"),
                api_key=os.environ.get("HUGGINGFACE_API_KEY")
            )

            messages = [
                {
                    "role": "system", 
                    "content": "You are a helpful assistant that outputs JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            chat_completion = client.chat.completions.create(
                model="LeeundEr/Qwen3-14B-Finetune",
                messages=messages,
                stream=True,
                max_tokens=4096,
                temperature=0.1
            )

            content = ""
            for message in chat_completion:
                if message.choices[0].delta.content:
                    content += message.choices[0].delta.content

        elif llm:
            messages = [
                SystemMessage(content="You are a helpful assistant that outputs JSON."),
                HumanMessage(content=prompt)
            ]
            result = llm.invoke(messages)
            content = result.content
                
        else:
            # Use OpenRouter
            llm = ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                model="qwen/qwen3-14b",
                temperature=0.1,
                max_tokens=4096
            )
            
            messages = [
                SystemMessage(content="You are a helpful assistant that outputs JSON."),
                HumanMessage(content=prompt)
            ]
            
            result = llm.invoke(messages)
            content = result.content

        print(f"DEBUG: Raw content from LLM (use_finetuned={use_finetuned}):\n{content}")
        
        if not content:
            raise ValueError("LLM returned empty content")

        cleaned_content = clean_json_text(content)
        print(f"DEBUG: Cleaned content: {repr(cleaned_content)}")
        
        try:
            data = json.loads(cleaned_content)
        except json.JSONDecodeError:
            print(f"Warning: JSON decode failed. Attempting fallback extraction.")
            data = extract_score_and_summary(content)
        
        # Ensure required fields
        if "score" not in data:
            # Try one more time to find just a number if everything else failed
            try:
                score = float(cleaned_content.strip())
                data["score"] = score
                data["summary"] = "Score extracted from raw number."
            except ValueError:
                data["score"] = 5.0
                
        if "summary" not in data:
            data["summary"] = "No summary provided."
            
        return data
        
    except Exception as e:
        print(f"Error in scoring (use_finetuned={use_finetuned}): {e}")
        
        return {
            "score": 5.0,
            "summary": f"Error calculating score: {str(e)}"
        }
