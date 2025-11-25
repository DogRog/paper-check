from unsloth import FastLanguageModel

# 1. Load your fine-tuned adapters
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "final_thesis_model/lora_adapters", # Point to where you saved it
    max_seq_length = 16384,
    dtype = None,
    load_in_4bit = True,
)

# 2. Convert to GGUF (Quantized format)
# q4_k_m is the standard balance of speed/quality
print("Converting to GGUF (This may take 5-10 mins)...")
model.save_pretrained_gguf("thesis_model_q4_k_m", tokenizer, quantization_method = "q4_k_m")

print("Export complete. You now have 'thesis_model_q4_k_m-unsloth.Q4_K_M.gguf'")