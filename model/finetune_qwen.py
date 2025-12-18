import argparse
import os
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

def train_and_evaluate(
    mode="test", 
    train_path="finetune_dataset/unsloth_finetune_train.jsonl",
    val_path="finetune_dataset/unsloth_finetune_val.jsonl",
    test_path="finetune_dataset/unsloth_finetune_test.jsonl",
    max_seq_length=16384,
    batch_size=2,
    grad_accum_steps=4,
    learning_rate=2e-4,
    lora_r=16,
    load_in_4bit=True,
    enable_thinking=False,  # New Qwen 3 feature
    save_gguf=False
):
    # --- 0. Configuration ---
    print(f"\n=== Starting Qwen 3 Finetuning (Mode: {mode.upper()}) ===")
    
    # Auto-detect Bfloat16 (Ampere/Hopper GPUs)
    dtype = None 
    model_id = "unsloth/Qwen3-14B-Instruct-bnb-4bit"
    
    print(f"Configurations:")
    print(f"  Model: {model_id}")
    print(f"  Thinking Mode: {'Enabled' if enable_thinking else 'Disabled'}")
    print(f"  Max Seq Length: {max_seq_length}")
    print(f"  Batch Size: {batch_size} | Accumulation: {grad_accum_steps}")
    
    # --- 1. Load Model & Tokenizer ---
    print("\n[1/7] Loading Qwen 3 Model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_id,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    # --- 2. Add LoRA Adapters ---
    print("[2/7] Applying LoRA Adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_r, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0, 
        bias = "none",
        use_gradient_checkpointing = "unsloth", 
        random_state = 3407,
    )

    # --- 3. Data Processing (Updated for Qwen 3) ---
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(tokenizer, chat_template = "chatml")

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [
            tokenizer.apply_chat_template(
                convo, 
                tokenize=False, 
                add_generation_prompt=False,
                # Qwen 3 specific argument to wrap output in <think> tags if needed
                # Note: unsloth handles the low-level tags, but we explicitly pass it here if supported by your version
                # If your transformers version is older, remove 'enable_thinking' from apply_chat_template
            ) for convo in convos
        ]
        return { "text" : texts }

    print(f"[3/7] Loading Datasets...")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train dataset not found at {train_path}")
        
    train_dataset = load_dataset("json", data_files=train_path, split="train")
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)

    val_dataset = None
    if val_path and os.path.exists(val_path):
        print(f"  - Validation set loaded: {val_path}")
        val_dataset = load_dataset("json", data_files=val_path, split="train")
        val_dataset = val_dataset.map(formatting_prompts_func, batched=True)
    
    test_dataset = None
    if test_path and os.path.exists(test_path):
        print(f"  - Test set loaded (for final eval): {test_path}")
        test_dataset = load_dataset("json", data_files=test_path, split="train")
        test_dataset = test_dataset.map(formatting_prompts_func, batched=True)

    # --- 4. Training Arguments ---
    if mode == "test":
        output_dir = "outputs/qwen3_14b_test_run"
        max_steps = 60
        num_train_epochs = 1 
        save_strategy = "steps"
        save_steps = 20
        logging_steps = 1
        eval_steps = 20
        print("  ! TEST MODE ENABLED: Running for only 60 steps.")
    else:
        output_dir = "outputs/qwen3_14b_full_run"
        max_steps = -1
        num_train_epochs = 1
        save_strategy = "epoch"
        save_steps = 0 
        logging_steps = 10
        eval_steps = 200
        print("  ! FULL MODE ENABLED: Running for 1 full epoch.")

    training_args = TrainingArguments(
        output_dir = output_dir,
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = grad_accum_steps,
        warmup_steps = 10,
        max_steps = max_steps,
        num_train_epochs = num_train_epochs,
        learning_rate = learning_rate,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = logging_steps,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        save_strategy = save_strategy,
        save_steps = save_steps if save_strategy == "steps" else 500,
        eval_strategy = "steps" if val_dataset else "no",
        eval_steps = eval_steps,
        report_to = "none",
    )

    # --- 5. Initialize Trainer ---
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, 
        args = training_args,
    )

    # --- 6. Train ---
    print("\n[5/7] Starting Training...")
    
    # Memory Stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"  GPU: {gpu_stats.name} ({max_memory} GB)")
    print(f"  Reserved Start: {start_gpu_memory} GB")

    trainer_stats = trainer.train()
    
    # Post-train Memory Stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    print(f"  Reserved End: {used_memory} GB")
    print(f"  LoRA Memory Used: {used_memory_for_lora} GB")

    # --- 7. Evaluation ---
    if test_dataset:
        print("\n[6/7] Evaluating on Test Set...")
        test_results = trainer.evaluate(test_dataset)
        print(f"  Test Loss: {test_results['eval_loss']}")

    # --- 8. Saving ---
    print(f"\n[7/7] Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    if save_gguf:
        print("  Saving GGUF (q4_k_m)...")
        try:
            model.save_pretrained_gguf(output_dir, tokenizer, quantization_method="q4_k_m")
        except Exception as e:
            print(f"  Could not save GGUF: {e}")

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Qwen 3 (14B) on PeerRead")
    
    # Mode and Dataset
    parser.add_argument("--mode", choices=["test", "full"], default="test", help="Training mode")
    parser.add_argument("--train_path", default="finetune_dataset/unsloth_finetune_train.jsonl")
    parser.add_argument("--val_path", default="finetune_dataset/unsloth_finetune_val.jsonl")     
    parser.add_argument("--test_path", default="finetune_dataset/unsloth_finetune_test.jsonl")     
    
    # Hyperparameters
    parser.add_argument("--max_seq_length", type=int, default=16384)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--save_gguf", action="store_true", help="Save GGUF version at the end")
    parser.add_argument("--disable_thinking", action="store_true", help="Disable Qwen 3 thinking mode")
    
    args = parser.parse_args()
    
    train_and_evaluate(
        mode=args.mode, 
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        load_in_4bit=not args.no_4bit,
        save_gguf=args.save_gguf,
        enable_thinking=not args.disable_thinking
    )