# 2. THE ALIGNMENT TAX BENCHMARK SCRIPT
import torch
import time
import re
import pandas as pd
from datasets import load_dataset
from unsloth import FastLanguageModel

# --- CONFIGURATION ---
MODEL_SFT = "nallaramu/deliberate-qwen-2.5-3b-reasoning"
MODEL_DPO = "nallaramu/deliberate-qwen-2.5-3b-dpo"
NUM_QUESTIONS = 50

# Load test dataset
print("Loading GSM8K Test Set...")
dataset = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=42).select(range(NUM_QUESTIONS))

def extract_answer(text):
    """Extracts the final answer from inside the <answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""

def check_compliance(text, num_tokens):
    """Checks if the model obeyed formatting and stopped naturally."""
    # 1. Check if it hit the rambling token limit
    if num_tokens >= 512:
        return False
    # 2. Check for hallucinated tags (e.g., <suggestion>)
    if "<suggestion>" in text.lower():
        return False
    # 3. Check if it properly closed the answer tag
    if "</answer>" not in text:
        return False
    return True

def run_evaluation(model_id):
    print(f"\n{'='*40}")
    print(f"EVALUATING: {model_id}")
    print(f"{'='*40}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_id,
        max_seq_length = 2048,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)
    
    correct_count = 0
    compliant_count = 0
    total_tokens = 0
    
    for i, item in enumerate(dataset):
        q = item['question']
        gt = item['answer'].split("####")[-1].strip()
        
        prompt = f"### Question:\n{q}\n\n### Reasoning:\n"
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        
        # We rely on the model's natural eos_token to stop
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.1, # Low temp for deterministic logic
            do_sample=False
        )
        
        prompt_len = inputs.input_ids.shape[1]
        new_tokens = outputs[0][prompt_len:]
        num_tokens = len(new_tokens)
        total_tokens += num_tokens
        
        full_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # 1. Check Compliance
        is_compliant = check_compliance(full_output, num_tokens)
        if is_compliant:
            compliant_count += 1
            
        # 2. Check Accuracy (Win Rate)
        model_ans = extract_answer(full_output)
        # Simple extraction check (if ground truth number is in the model's answer box)
        if gt in model_ans:
            correct_count += 1
            
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{NUM_QUESTIONS}...")

    # Cleanup VRAM for the next model
    del model, tokenizer
    torch.cuda.empty_cache()
    
    metrics = {
        "Accuracy (Win Rate)": (correct_count / NUM_QUESTIONS) * 100,
        "Format Compliance": (compliant_count / NUM_QUESTIONS) * 100,
        "Avg Tokens / Query": total_tokens / NUM_QUESTIONS
    }
    return metrics

# --- EXECUTE THE BENCHMARK ---
metrics_sft = run_evaluation(MODEL_SFT)
metrics_dpo = run_evaluation(MODEL_DPO)

# --- DISPLAY THE RESULTS ---
print("\n\n" + "*"*60)
print("🏆 ALIGNMENT TAX BENCHMARK RESULTS 🏆")
print("*"*60)

results_table = {
    "Metric": ["Accuracy (Win Rate)", "Format Compliance", "Avg Verbosity (Tokens)"],
    "SFT Base Model (Project 1)": [
        f"{metrics_sft['Accuracy (Win Rate)']:.1f}%",
        f"{metrics_sft['Format Compliance']:.1f}%",
        f"{metrics_sft['Avg Tokens / Query']:.0f} tokens"
    ],
    "DPO Aligned Model (Project 2)": [
        f"{metrics_dpo['Accuracy (Win Rate)']:.1f}%",
        f"{metrics_dpo['Format Compliance']:.1f}%",
        f"{metrics_dpo['Avg Tokens / Query']:.0f} tokens"
    ],
    "Impact (Alignment Tax/Gain)": [
        f"{metrics_dpo['Accuracy (Win Rate)'] - metrics_sft['Accuracy (Win Rate)']:.1f}%",
        f"{metrics_dpo['Format Compliance'] - metrics_sft['Format Compliance']:+.1f}%",
        f"{metrics_dpo['Avg Tokens / Query'] - metrics_sft['Avg Tokens / Query']:+.0f} tokens"
    ]
}

df = pd.DataFrame(results_table)
print("\n" + df.to_markdown(index=False))