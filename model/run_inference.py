from unsloth import FastLanguageModel
import torch

# --- Configuration ---
# Point this to your saved output folder
model_path = "outputs/checkpoint-100" 
max_seq_length = 16384 # Keep consistent with training
dtype = None
load_in_4bit = True

# --- 1. Load the Fine-Tuned Model ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Enable native 2x faster inference
FastLanguageModel.for_inference(model) 

# --- 2. Define a Test Paper (Snippet) ---
# We use a paper NOT in the training set (or a made up one) to test generalization.
paper_text = """
TITLE: "Optimizing LLMs via Telepathic Prompting"

## 1 INTRODUCTION
We propose a novel method where the user thinks about the prompt and the LLM retrieves it via Wi-Fi signals.
Existing methods require typing, which is slow. Our method, TelePrompt, uses SOTA brain-computer interfaces.

## 4 EXPERIMENTS
We tested this on 3 users. Accuracy was 12%. We believe this is due to atmospheric interference.
"""

# --- 3. Format the Prompt ---
messages = [
    {"role": "system", "content": "You are an expert Area Chair for a top-tier AI conference. Critique the following paper based on: Novelty, Soundness, and Significance."},
    {"role": "user", "content": f"TITLE: Optimizing LLMs via Telepathic Prompting\n\nCONTENT:\n{paper_text}\n\n---\nTask: Provide a detailed peer review and a final decision."},
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Must be True for generation
    return_tensors = "pt",
).to("cuda")

# --- 4. Generate Review ---
print("Generating Review (this may take 10-20 seconds)...")

outputs = model.generate(
    input_ids = inputs,
    max_new_tokens = 1024, # Allow it to write a long review
    use_cache = True,
    temperature = 0.7,
)

# Decode and print only the new text
response = tokenizer.batch_decode(outputs)
print("\n--- GENERATED REVIEW ---\n")
print(response[0].split("<|im_start|>assistant")[-1].replace("<|im_end|>", ""))