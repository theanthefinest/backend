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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.set_per_process_memory_fraction(0.85)
torch.cuda.empty_cache()
gc.collect()

config = configparser.ConfigParser()
config.read('configuration.ini')

logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)

log.info("Load the model: ...")

token = config.get('token', 'token_key')
login(token)

BASE_MODEL = config.get('Model', 'base_model')
ADAPTER_MODEL = "/home/chhaythean-ly/backend/Driver/checkpoint-750"

log.info(" Tokenizer loading ... ")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

log.info("Success...!")

log.info("load the model....")

quant_config = BitsAndBytesConfig(load_in_4bit=True, llm_int8_enable_fp32_cpu_offload=True)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto",
)

model = PeftModel.from_pretrained(model, ADAPTER_MODEL)
model.eval()

log.info("Model loaded and ready for inference.")

app = FastAPI()

class RequestBody(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

model_device = next(model.parameters()).device
@app.post("/chat")
async def chat(req: RequestBody):
    log.info(f"Received prompt: {req.prompt[:50]}...")
    inputs = tokenizer(req.prompt, return_tensors="pt", padding=True, truncation=True).to(model_device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    log.info(f"Generated response: {response[:50]}...")
    return {'response': response}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.on_event("startup")
async def warmup_model():
    log.info("Warming up model...")
    inputs = tokenizer("Hello", return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=5)
    log.info("Model warm-up done.")
