from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from langchain_router import *
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain, PromptTemplate

# ========= API For LangChain ChatBot ========>

app = FastAPI()
port = 6969

class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    response: str

smart_llm = Router()
memory = ConversationBufferMemory(input_key="input", memory_key="history")

prompt_template = PromptTemplate(
    input_variables=["history", "input"],
    template="""
Previous conversation:
{history}

Human: {input}
Assistant:""",
)

conversation_chain = LLMChain(
    llm=smart_llm,
    prompt=prompt_template,
    memory=memory,
    verbose=True,
)

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        response = conversation_chain.run(input=request.prompt)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/health")
async def health_check():
    return {"status": "ok"}