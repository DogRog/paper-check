from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# --- Configuration ---
max_seq_length = 16384 
dtype = None 
load_in_4bit = True 

# --- Load Model & Data ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
)

# Chat Template
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml",
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"},
)

# Load your JSONL
dataset = load_dataset("json", data_files="peerread_qwen_fulltext.jsonl", split="train")

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts }

dataset = dataset.map(formatting_prompts_func, batched = True)

# --- THE FULL TRAINING CONFIG ---
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, 
        
        # --- CHANGES FOR FULL RUN ---
        num_train_epochs = 1,      # Train on the whole dataset once
        max_steps = -1,            # Disable step limit
        save_strategy = "epoch",   # Save at the end of the epoch
        logging_steps = 10,        # Log less frequently to keep output clean
        # ----------------------------
        
        learning_rate = 2e-4,
        fp16 = False,
        bf16 = True, 
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "final_thesis_model",
    ),
)

print("Starting Full Training (Approx 1-2 hours on RTX 4090)...")
trainer.train()

# --- SAVE EVERYTHING ---
print("Saving adapters...")
model.save_pretrained("final_thesis_model/lora_adapters")
tokenizer.save_pretrained("final_thesis_model/lora_adapters")
print("Done.")