import argparse
import os
import torch
from unsloth import FastLanguageModel

def export_model(
    model_path,
    hub_repo,
    token=None,
    push_mode="merged_16bit",
    max_seq_length=32768, 
    quantization_method="q4_k_m" 
):
    print(f"\n=== Starting Model Export ===")
    print(f"  Input Model Path: {model_path}")
    print(f"  Target Repo: {hub_repo}")
    print(f"  Mode: {push_mode}")
    print(f"  Max Seq Length: {max_seq_length}")

    # --- 1. Load the Fine-Tuned Model ---
    print(f"\n[1/3] Loading local model from {model_path}...")
    
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_path,
            max_seq_length = max_seq_length,
            dtype = None,
            load_in_4bit = True, 
        )
    except OSError:
        print(f"Error: Could not find model at '{model_path}'. Check your path.")
        return

    # --- 2. Process and Push ---
    print(f"\n[2/3] Processing and Pushing to Hugging Face Hub...")
    
    if token is None:
        token = os.getenv("HF_TOKEN")
        
    if push_mode == "lora":
        print("  -> Pushing LoRA adapters only...")
        model.push_to_hub(hub_repo, token=token)
        tokenizer.push_to_hub(hub_repo, token=token)

    elif push_mode == "merged_16bit":
        print("  -> Merging to 16-bit (Best for generic usage)...")
        model.push_to_hub_merged(
            hub_repo, 
            tokenizer, 
            save_method="merged_16bit", 
            token=token
        )

    elif push_mode == "merged_4bit":
        print("  -> Merging to 4-bit (Unsloth/BnB specific)...")
        model.push_to_hub_merged(
            hub_repo, 
            tokenizer, 
            save_method="merged_4bit", 
            token=token
        )

    elif push_mode == "gguf":
        print(f"  -> Converting and pushing GGUF ({quantization_method})...")
        model.push_to_hub_gguf(
            hub_repo, 
            tokenizer, 
            quantization_method=quantization_method, 
            token=token
        )
        
    else:
        print(f"Unknown mode: {push_mode}")
        return

    print("\n[3/3] Done! Check your repository at:")
    print(f"https://huggingface.co/{hub_repo}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Unsloth Finetune to Hugging Face")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to your local output directory")
    parser.add_argument("--hub_repo", type=str, required=True, help="Target Hugging Face Repo")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face Write Token")
    parser.add_argument("--mode", type=str, default="merged_16bit", choices=["lora", "merged_16bit", "merged_4bit", "gguf"])
    parser.add_argument("--max_seq_length", type=int, default=32768, help="Must match training seq length")
    parser.add_argument("--quant_method", type=str, default="q4_k_m")

    args = parser.parse_args()
    
    export_model(
        model_path=args.model_path,
        hub_repo=args.hub_repo,
        token=args.token,
        push_mode=args.mode,
        max_seq_length=args.max_seq_length,
        quantization_method=args.quant_method
    )