from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# --- Configuration ---
DATASET_FILE = "peerread_qwen_fulltext.jsonl"
OUTPUT_DIR = "outputs"

# --- Configuration for RTX 4090 ---
# 24GB VRAM allows us to push max_seq_length higher.
# Start with 16384 (16k). If you get OOM (Out of Memory), drop to 8192.
max_seq_length = 16384 
dtype = None # Auto-detects Bfloat16 (Native to 4090 - faster/more stable)
load_in_4bit = True # Mandatory for 14B model on 24GB card

# --- 1. Load the 14B Model ---
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-14B-Instruct-bnb-4bit", # Pre-quantized 4-bit
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# --- 2. Add LoRA Adapters ---
print("Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",
    use_gradient_checkpointing = "unsloth", # CRITICAL for 4090 memory savings
    random_state = 3407,
)

# --- 3. Chat Template ---
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml",
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"},
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts }

# --- 4. Load and Format Dataset ---
print(f"Loading dataset from {DATASET_FILE}...")
dataset = load_dataset("json", data_files=DATASET_FILE, split="train")
dataset = dataset.map(formatting_prompts_func, batched = True)

# --- 5. Trainer (Tuned for 4090) ---
print("Starting training...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2, # Start here. If VRAM is free, try 4.
        gradient_accumulation_steps = 4, # Effective batch size = 2 * 4 = 8
        warmup_steps = 10,
        max_steps = 100, # Adjust based on dataset size
        learning_rate = 2e-4,
        fp16 = False, # 4090 supports BF16, which is better
        bf16 = True,  # Enable BF16
        logging_steps = 1,
        optim = "adamw_8bit", # Saves VRAM
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = OUTPUT_DIR,
    ),
)

trainer.train()
print("Training complete!")
