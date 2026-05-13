# ==========================================
# 2. DPO TRAINING & W&B TRACKING
# ==========================================
import torch
import wandb
import os
from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
from huggingface_hub import notebook_login

# --- AUTHENTICATION ---
print("Log in to Weights & Biases:")
wandb.login() # Prompts for W&B API Key

print("\nLog in to Hugging Face (Must have WRITE access):")
notebook_login() # Prompts for Hugging Face Token

# Initialize W&B Project
wandb.init(project="dpo-alignment-flywheel", name="qwen-3b-length-debiased-dpo")

# --- CONFIGURATION ---
MODEL_ID = "nallaramu/deliberate-qwen-2.5-3b-reasoning"
HF_PUSH_ID = "nallaramu/deliberate-qwen-2.5-3b-dpo" # Where the final model goes

print(f"Loading Base SFT Model: {MODEL_ID} in 4-bit...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_ID,
    max_seq_length = 2048,
    load_in_4bit = True,
)

# Apply LoRA Adapters for the Active DPO Model
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# --- LOAD DATASET ---
if not os.path.exists("dpo_preference_pairs.jsonl"):
    raise FileNotFoundError("CRITICAL: Please upload 'dpo_preference_pairs.jsonl' to Colab!")

# Load the 74 length-debiased preference pairs
dataset = load_dataset("json", data_files="dpo_preference_pairs.jsonl", split="train")

# --- UNSLOTH DPO OPTIMIZATION ---
# This patches the trainer to avoid the 2x VRAM Reference Model requirement
PatchDPOTrainer()

# --- DPO CONFIGURATION ---
dpo_args = DPOConfig(
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    warmup_steps = 5,
    max_steps = 50, # 50 steps is enough for our 74 high-quality pairs
    learning_rate = 5e-5, # Lower LR than SFT to prevent representation collapse
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 1,
    optim = "adamw_8bit",
    output_dir = "dpo_outputs",
    
    # W&B Integration
    report_to = "wandb", 
    
    # DPO Specifics
    beta = 0.1, # The KL penalty. 0.1 allows behavior change without destroying logic.
    max_length = 2048,
    max_prompt_length = 1024,
    remove_unused_columns = False # Required so 'prompt', 'chosen', 'rejected' map correctly
)

print("Initializing DPOTrainer...")
dpo_trainer = DPOTrainer(
    model = model,
    ref_model = None, # Unsloth dynamically handles the reference model
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = dpo_args,
)

print("Starting Direct Preference Optimization...")
dpo_trainer.train()

# --- SAVE & PUSH ALIGNED MODEL ---
print("Training Complete! Pushing to Hugging Face...")
# We save locally first
model.save_pretrained("dpo_aligned_adapter")
tokenizer.save_pretrained("dpo_aligned_adapter")

# Push to Hub with professional tags
model.push_to_hub(
    HF_PUSH_ID,
    tags=["dpo", "rlhf", "reasoning", "unsloth", "alignment"],
    commit_message="Initial release of DPO-aligned debiased reasoning model"
)
tokenizer.push_to_hub(HF_PUSH_ID)

wandb.finish()
print(f"🚀 Success! Model live at: https://huggingface.co/{HF_PUSH_ID}")