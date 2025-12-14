import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import re

# Global model and tokenizer to avoid reloading
_model = None
_tokenizer = None

MODEL_PATH = "../model/outputs/qwen_peerread" # Path to the fine-tuned adapter
BASE_MODEL_ID = "unsloth/Qwen3-14B-Instruct-bnb-4bit" # Base model

def load_local_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    print(f"Loading local model from {MODEL_PATH}...")
    
    try:
        # Check if adapter exists
        if not os.path.exists(MODEL_PATH):
            print(f"Warning: Adapter not found at {MODEL_PATH}. Using base model only.")
            adapter_path = None
        else:
            adapter_path = MODEL_PATH

        # Load Tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

        # Load Model
        # Note: load_in_4bit=True requires bitsandbytes and CUDA. 
        # If on Mac/CPU, this might fail or need different settings.
        # We try to detect if CUDA is available.
        if torch.cuda.is_available():
            _model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID,
                device_map="auto",
                load_in_4bit=True,
                trust_remote_code=True
            )
        else:
            # Fallback for non-CUDA (e.g. Mac MPS) - might be slow or OOM for 14B
            print("CUDA not detected. Loading in float16 (requires ~28GB RAM)...")
            _model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )

        if adapter_path:
            print("Loading LoRA adapter...")
            _model = PeftModel.from_pretrained(_model, adapter_path)
            
        print("Model loaded successfully.")
        return _model, _tokenizer

    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def score_paper_with_local_model(paper_text: str, issues_summary: str) -> dict:
    """
    Scores the paper using the locally fine-tuned Qwen model.
    """
    model, tokenizer = load_local_model()
    
    if not model or not tokenizer:
        return {
            "score": 5,
            "summary": "Error: Local model could not be loaded. Returning default score."
        }

    # Construct prompt similar to training data
    prompt = f"""TITLE: Paper Review
    
CONTENT:
{paper_text[:4000]} # Truncate to fit context if needed

ISSUES FOUND:
{issues_summary}

---
Task: Provide a detailed peer review and a final decision score (1-10).
"""
    
    messages = [
        {"role": "system", "content": "You are an expert Area Chair for a top-tier AI conference. Critique the paper and provide a score."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Parse the response to extract score and summary
    # Qwen3 might include <think> blocks. We can strip them for the summary if desired,
    # or keep them. For now, we'll try to clean it up if it's distinct.
    
    summary = response
    # Remove thinking block for cleaner summary if present (optional)
    summary = re.sub(r"<think>.*?</think>", "", summary, flags=re.DOTALL).strip()
    
    score = 5
    
    score_match = re.search(r"Score:\s*(\d+)", response, re.IGNORECASE)
    if score_match:
        try:
            score = int(score_match.group(1))
            score = max(1, min(10, score))
        except:
            pass
            
    return {
        "score": score,
        "summary": summary
    }
