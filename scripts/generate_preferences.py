# 2. THE FLYWHEEL SCRIPT (Run this in the next Colab cell)
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from openai import OpenAI
import json
import time

# --- CONFIGURATION ---
MODEL_ID = "nallaramu/deliberate-qwen-2.5-3b-reasoning"
GROQ_API_KEY = "YOUR_GROQ_API_KEY" # Insert your Groq key here
OUTPUT_FILE = "dpo_preference_pairs.jsonl"
NUM_SAMPLES = 200 # 200 pairs is enough for a strong DPO signal on a 3B model

# Initialize Groq Client
client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

# Load Student Model for Generation
print(f"Loading {MODEL_ID}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# Load Unseen Data (GSM8K Test Split)
dataset = load_dataset("openai/gsm8k", "main", split="test")

# --- THE JUDGE PROMPT ---
JUDGE_SYSTEM_PROMPT = """You are an expert AI Alignment Judge. 
You will be given a math question, the Ground Truth answer, and two reasoning paths (Response A and Response B).
Your job is to pick the BEST response based on:
1. Logical correctness (Does it reach the Ground Truth?).
2. Conciseness (Is it direct and free of repetitive rambling?).
3. Formatting (Does it properly use <think> and </answer> tags without trailing garbage?).

Output ONLY the letter "A" or "B". Do not output any other text."""

def generate_two_paths(question):
    prompt = f"### Question:\n{question}\n\n### Reasoning:\n"
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    # We use do_sample=True and temperature=0.8 to force the model to think differently each time
    paths =[]
    for _ in range(2):
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512, 
            use_cache=True, 
            do_sample=True, 
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        paths.append(tokenizer.decode(new_tokens, skip_special_tokens=False))
    return paths[0], paths[1]

def judge_responses(question, gt, resp_a, resp_b):
    user_content = f"Question: {question}\nGround Truth: {gt}\n\n--- Response A ---\n{resp_a}\n\n--- Response B ---\n{resp_b}"
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0,
            max_tokens=5
        )
        winner = response.choices[0].message.content.strip().upper()
        return winner if winner in ["A", "B"] else None
    except Exception as e:
        print(f"Groq API Error: {e}")
        return None

# --- THE FLYWHEEL LOOP ---
print("\nStarting the Data Flywheel...")
preference_data =[]
success_count = 0

for i, item in enumerate(dataset):
    if success_count >= NUM_SAMPLES:
        break
        
    question = item['question']
    gt_answer = item['answer'].split("####")[-1].strip()
    
    print(f"\n[{success_count+1}/{NUM_SAMPLES}] Processing question...")
    
    # 1. Generate candidate paths
    path_a, path_b = generate_two_paths(question)
    
    # 2. Ask the Judge
    winner = judge_responses(question, gt_answer, path_a, path_b)
    time.sleep(2) # Respect API rate limits
    
    if not winner:
        continue
        
    chosen = path_a if winner == "A" else path_b
    rejected = path_b if winner == "A" else path_a
    
    # 3. LENGTH-DEBIASING FILTER (The "Expert Signal")
    # We only keep the pair if the Chosen response is shorter or equal to the Rejected one
    chosen_len = len(tokenizer.encode(chosen))
    rejected_len = len(tokenizer.encode(rejected))
    
    if chosen_len > rejected_len:
        print(f"   [Filtered] Chosen path was longer ({chosen_len} vs {rejected_len} tokens). Discarding to prevent DPO length bias.")
        continue
        
    print(f"   [Kept] Chosen: {chosen_len} tokens | Rejected: {rejected_len} tokens.")
    
    # 4. Format for Hugging Face DPOTrainer
    # DPO requires the exact prompt that preceded the chosen/rejected texts
    dpo_prompt = f"### Question:\n{question}\n\n### Reasoning:\n"
    
    pair = {
        "prompt": dpo_prompt,
        "chosen": chosen,
        "rejected": rejected
    }
    
    # Save incrementally
    with open(OUTPUT_FILE, "a") as f:
        f.write(json.dumps(pair) + "\n")
        
    preference_data.append(pair)
    success_count += 1

print(f"\n🎉 Flywheel Complete! Saved {len(preference_data)} length-debiased preference pairs to {OUTPUT_FILE}")