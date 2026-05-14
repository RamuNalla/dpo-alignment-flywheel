import streamlit as st
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(page_title="Alignment Arena | DPO vs SFT", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for an "Elite Engineering" look
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .model-box { background-color: #1A1C23; padding: 20px; border-radius: 10px; border: 1px solid #2D303E; height: 500px; overflow-y: auto; font-family: monospace; }
    .metric-container { display: flex; justify-content: space-between; margin-top: 10px; padding: 10px; background-color: #12141A; border-radius: 8px; border: 1px solid #2D303E; }
    .metric-item { text-align: center; }
    .metric-value { font-size: 20px; font-weight: bold; color: #00E676; }
    .metric-label { font-size: 12px; color: #8892B0; text-transform: uppercase; }
    .header-text { text-align: center; margin-bottom: 30px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='header-text'>⚔️ The Alignment Arena</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8892B0;'>Benchmarking Base SFT Reasoning vs. Length-Debiased DPO Alignment</p>", unsafe_allow_html=True)

# ==========================================
# 2. MODEL LOADING (Cached)
# ==========================================
@st.cache_resource(show_spinner="Loading Models into Mac Unified Memory...")
def load_models():
    SFT_ID = "nallaramu/deliberate-qwen-2.5-3b-reasoning"
    DPO_ID = "nallaramu/deliberate-qwen-2.5-3b-dpo"
    
    tokenizer = AutoTokenizer.from_pretrained(SFT_ID)
    
    # MAC FIX: Removed load_in_4bit, using torch.float16. 
    # device_map="auto" will automatically use Apple's 'mps' (Metal Performance Shaders)
    sft_model = AutoModelForCausalLM.from_pretrained(SFT_ID, torch_dtype=torch.float16, device_map="auto")
    dpo_model = AutoModelForCausalLM.from_pretrained(DPO_ID, torch_dtype=torch.float16, device_map="auto")
    
    return tokenizer, sft_model, dpo_model

tokenizer, sft_model, dpo_model = load_models()

# ==========================================
# 3. UI LAYOUT
# ==========================================
query = st.text_input("Enter a Logic/Math Prompt:", value="If I have 12 apples, give half to a friend, and buy 4 more, how many do I have?")
start_btn = st.button("🚀 Run Alignment Benchmark", use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🤖 Base SFT Model (Project 1)")
    st.caption("Prone to over-explaining and length-bias.")
    sft_text_box = st.empty()
    sft_metrics = st.empty()
    
with col2:
    st.markdown("### 🎯 DPO Aligned Model (Project 2)")
    st.caption("RLAIF trained for extreme conciseness and accuracy.")
    dpo_text_box = st.empty()
    dpo_metrics = st.empty()

# Setup initial UI states
sft_text_box.markdown("<div class='model-box'>Waiting for prompt...</div>", unsafe_allow_html=True)
dpo_text_box.markdown("<div class='model-box'>Waiting for prompt...</div>", unsafe_allow_html=True)

# ==========================================
# 4. GENERATION ENGINE
# ==========================================
def run_generation(model, placeholder, metrics_placeholder):
    prompt = f"### Question:\n{query}\n\n### Reasoning:\n"
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Start thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    generated_text = ""
    start_time = time.time()
    tokens = 0
    
    # Stream text and update metrics live
    for new_text in streamer:
        generated_text += new_text
        tokens += 1
        elapsed = time.time() - start_time
        tok_sec = tokens / elapsed if elapsed > 0 else 0
        
        # Color the <think> and <answer> tags for better visibility
        display_text = generated_text.replace("<think>", "🔄 **<think>**\n").replace("</think>", "\n**</think>**").replace("<answer>", "✅ **<answer>**\n")
        
        placeholder.markdown(f"<div class='model-box'>{display_text}</div>", unsafe_allow_html=True)
        
        # Update Live Metrics
        metrics_placeholder.markdown(f"""
            <div class='metric-container'>
                <div class='metric-item'><div class='metric-value'>{elapsed:.1f}s</div><div class='metric-label'>Latency</div></div>
                <div class='metric-item'><div class='metric-value'>{tokens}</div><div class='metric-label'>Tokens</div></div>
                <div class='metric-item'><div class='metric-value'>{tok_sec:.1f}</div><div class='metric-label'>Tok/s</div></div>
            </div>
        """, unsafe_allow_html=True)
        
    thread.join()

# ==========================================
# 5. EXECUTION TRIGGER
# ==========================================
if start_btn:
    # Run sequentially to prevent CUDA out-of-memory errors and allow the user to watch the contrast
    with st.spinner("Generating SFT Base Response..."):
        run_generation(sft_model, sft_text_box, sft_metrics)
        
    with st.spinner("Generating DPO Aligned Response..."):
        run_generation(dpo_model, dpo_text_box, dpo_metrics)
        
    st.success("✅ Benchmark Complete! Notice the token reduction and strict format compliance in the DPO model.")