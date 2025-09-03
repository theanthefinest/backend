import os 
import torch
import gc 

def get_device():
    torch.cuda.set_per_process_memory_fraction(0.85)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")

if __name__ == "__main__":
    get_device()