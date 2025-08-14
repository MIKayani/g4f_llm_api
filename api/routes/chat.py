import time
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Mapping
import g4f
from langchain_core.language_models.llms import LLM
from core.config import get_models_list, get_models_providers

router = APIRouter()

# --- Custom G4F LLM with Fallback ---

class G4FLLM(LLM):
    model_type: str
    temperature: float = 0.7

    @property
    def _llm_type(self) -> str:
        return "g4f-llm-fallback"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        models_list = get_models_list()
        if self.model_type not in models_list:
            raise HTTPException(status_code=404, detail=f"Model type not found: {self.model_type}")

        model_names_to_try = models_list[self.model_type]
        models_providers = get_models_providers()

        for model_name in model_names_to_try:
            providers = models_providers.get(model_name, [])
            for provider_name in providers:
                try:
                    response = g4f.ChatCompletion.create(
                        model=model_name,
                        provider=getattr(g4f.Provider, provider_name),
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                    )
                    if response:
                        return str(response)
                except Exception as e:
                    print(f"Error with model {model_name} and provider {provider_name}: {e}")
                    continue
        
        raise HTTPException(status_code=500, detail="All providers for the requested model type failed.")

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_type": self.model_type, "temperature": self.temperature}

# --- Pydantic Models for Endpoints ---

class InvokeRequest(BaseModel):
    model_type: str
    prompt: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage = Field(default_factory=Usage)

# --- API Endpoints ---

@router.get("/models")
def list_models():
    models_list = get_models_list()
    return list(models_list.keys())

@router.post("/llm/invoke")
def invoke_llm(request: InvokeRequest):
    try:
        llm = G4FLLM(model_type=request.model_type)
        response = llm.invoke(request.prompt)
        return {"response": response}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    models_list = get_models_list()
    models_providers = get_models_providers()
    
    model_type = request.model

    if model_type not in models_list:
        raise HTTPException(status_code=404, detail=f"Model type not found: {model_type}")

    model_names_to_try = models_list[model_type]

    for model_name in model_names_to_try:
        providers = models_providers.get(model_name, [])
        for provider_name in providers:
            try:
                response_content = await g4f.ChatCompletion.create_async(
                    model=model_name,
                    provider=getattr(g4f.Provider, provider_name),
                    messages=[msg.dict() for msg in request.messages],
                    temperature=request.temperature,
                )

                if response_content:
                    assistant_message = ChatMessage(role="assistant", content=str(response_content))
                    choice = ChatCompletionChoice(index=0, message=assistant_message)
                    return ChatCompletionResponse(
                        model=model_name, 
                        choices=[choice]
                    )

            except Exception as e:
                print(f"Error with model {model_name} and provider {provider_name}: {e}")
                continue

    raise HTTPException(status_code=500, detail="All providers for the requested model type failed.")
