import argparse
import os
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

def train(
    mode="test", 
    dataset_path="peerread_qwen_fulltext.jsonl",
    max_seq_length=16384,
    batch_size=2,
    grad_accum_steps=4,
    learning_rate=2e-4,
    lora_r=16,
    load_in_4bit=True
):
    # --- Configuration ---
    dtype = None # Auto-detects Bfloat16
    
    print(f"--- Starting Qwen3 Finetuning in {mode.upper()} mode ---")
    print(f"Configuration:")
    print(f"  Max Seq Length: {max_seq_length}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Grad Accum Steps: {grad_accum_steps}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  LoRA Rank: {lora_r}")
    print(f"  4-bit Quantization: {load_in_4bit}")

    # --- 1. Load the Model ---
    print("Loading Qwen3-14B-Instruct...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen3-14B-unsloth-bnb-4bit",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    # --- 2. Add LoRA Adapters ---
    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_r, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = lora_r,
        lora_dropout = 0, 
        bias = "none",
        use_gradient_checkpointing = "unsloth", 
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

    # --- 4. Load Dataset ---
    print(f"Loading dataset from {dataset_path}...")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.map(formatting_prompts_func, batched = True)

    # --- 5. Training Arguments ---
    if mode == "test":
        output_dir = "outputs/qwen_peerread_test"
        max_steps = 100
        num_train_epochs = 1 
        save_strategy = "steps"
        save_steps = 50
        logging_steps = 1
        print("Configured for TEST run (100 steps).")
    else:
        output_dir = "outputs/qwen_peerread_full"
        max_steps = -1
        num_train_epochs = 1
        save_strategy = "epoch"
        save_steps = 0 # Ignored for epoch strategy
        logging_steps = 10
        print("Configured for FULL run (1 epoch).")

    training_args = TrainingArguments(
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = grad_accum_steps,
        warmup_steps = 10,
        max_steps = max_steps,
        num_train_epochs = num_train_epochs,
        learning_rate = learning_rate,
        fp16 = False,
        bf16 = True,
        logging_steps = logging_steps,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = output_dir,
        save_strategy = save_strategy,
        save_steps = save_steps if save_strategy == "steps" else 500,
    )

    # --- 6. Trainer ---
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        args = training_args,
    )

    # --- 7. Train ---
    print("Starting training...")
    trainer.train()
    
    # --- 8. Save ---
    print(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Qwen3 on PeerRead")
    
    # Mode and Dataset
    parser.add_argument("--mode", choices=["test", "full"], default="test", help="Training mode")
    parser.add_argument("--dataset", default="peerread_qwen_fulltext.jsonl", help="Path to dataset file")
    
    # Hyperparameters
    parser.add_argument("--max_seq_length", type=int, default=16384, help="Max sequence length (default: 16384)")
    parser.add_argument("--batch_size", type=int, default=2, help="Per device train batch size (default: 2)")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank (default: 16)")
    
    # Hardware/Optimization
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization (use 16-bit)")
    
    args = parser.parse_args()
    
    train(
        mode=args.mode, 
        dataset_path=args.dataset,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        load_in_4bit=not args.no_4bit
    )
