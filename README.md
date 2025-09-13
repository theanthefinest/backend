
# CamTour-AI Backend

This backend contains both a Python (FastAPI, LangChain) and a Node.js (Express) server for AI chatbot and API routing.

---

## Requirements

### Python (FastAPI, LangChain)

- Python 3.10+
- CUDA-enabled GPU (for best performance)
- [PyTorch](https://pytorch.org/) with CUDA support
- The following Python packages (see `requirements.txt`):
    - bitsandbytes
    - transformers
    - peft
    - fastapi
    - pydantic
    - uvicorn
    - protobuf
    - sentencepiece

### Node.js (Express Gateway)

- Node.js 18+
- The following npm packages (see `package.json`):
    - express
    - axios
    - cors
    - zod

---

## Setup Instructions

### 1. Python Backend (FastAPI & LangChain)

#### a. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

#### b. Install PyTorch with CUDA (if using GPU)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

#### c. Install Python requirements
```bash
pip install -r requirements.txt
```

#### d. (Optional) If you have issues with `transformers`:
```bash
pip install -U transformers
```

#### e. Run FastAPI server
```bash
cd fastapi
uvicorn main:app --host 0.0.0.0 --port 8000
```

#### f. Run LangChain server
```bash
cd ../langchain
uvicorn main:app --host 0.0.0.0 --port 6969
```

---

### 2. Node.js Express Gateway

#### a. Install dependencies
```bash
npm install
```

#### b. Run Express server
```bash
npx ts-node express.ts
```

---

## Project Structure

- `fastapi/` — Main FastAPI backend for model inference
- `langchain/` — LangChain-based chatbot API
- `express.ts` — Node.js Express gateway (TypeScript)
- `requirements.txt` — Python dependencies
- `package.json` — Node.js dependencies

---

## Notes

- Ensure your CUDA drivers are installed for GPU acceleration.
- Update `configuration.ini` in `fastapi/` with your HuggingFace token and model paths.
- For troubleshooting, check logs in each service's terminal.

---

## License

See [LICENSE](LICENSE) for details.