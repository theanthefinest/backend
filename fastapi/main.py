# ======= FastAPI ========>

from fastapi import FastAPI, Request, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel 
import torch
import logging
import configparser 
from device import get_device
from huggingface_hub import hf_hub_download, login
import os 
import gc

# ====== Enable Torch CUDA ======>

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.set_per_process_memory_fraction(0.85)
torch.cuda.empty_cache()
gc.collect()

# ====== Load the Configuration ======>
config = configparser.ConfigParser()
config.read('configuration.ini')

# ====== Load Info with Color ========>
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"
RESET = "\x1b[0m"
# ====== End ========>

token = config.get('token', 'token_key')
login(token)

BASE_MODEL = config.get('Model', 'base_model')
ADAPTER_MODEL = "/media/chhay-thean/Drive D/CamTour-Ai/backend/fastapi/driver/chatbot_v0.3"

print(f"{GREEN}INFO{RESET}:     Loading The Tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

print(f"{GREEN}INFO{RESET}:     Tokenizer Loaded Successfully.")


print(f"{GREEN}INFO{RESET}:     Loading The Model...")
quant_config = BitsAndBytesConfig(load_in_4bit=True, llm_int8_enable_fp32_cpu_offload=True)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto",
)

model = PeftModel.from_pretrained(model, ADAPTER_MODEL)
model.eval()

print(f"{GREEN}INFO{RESET}:     Model Loaded Successfully.")

app = FastAPI()

class RequestBody(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.3
    top_p: Optional[float] = 0.9

model_device = next(model.parameters()).device

@app.post("/chat")
async def chat(req: RequestBody):
    log.info(f"Received prompt: {req.prompt[:50]}...")

    messages = [
        {"role": "user", "content": req.prompt}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model_device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"], 
            max_new_tokens=min(req.max_new_tokens, 128),  
            temperature=max(0.1, req.temperature),
            top_p=req.top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)

    if "[/INST]" in full_response:
        assistant_response = full_response.split("[/INST]")[1].strip()
    elif "### Response:" in full_response:
        assistant_response = full_response.split("### Response:")[1].strip()
    else:
        decoded_input = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
        assistant_response = full_response[len(decoded_input):].strip()
    # if '.' in assistant_response:
    #     assistant_response = assistant_response.split('.')[0].strip() + '.'
    # else:
    #     assistant_response = assistant_response.split('\n')[0].strip() 

    log.info(f"Generated response: {assistant_response[:50]}...")
    return {'response': assistant_response}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# @app.on_event("startup")
# async def warmup_model():
#     log.info("Warming up model...")
#     inputs = tokenizer("Hello", return_tensors="pt").to(device)
#     with torch.no_grad():
#         _ = model.generate(**inputs, max_new_tokens=5)
#     log.info("Model warm-up done.")
