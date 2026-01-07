import sglang as sgl
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from sglang import gen

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    stop: Optional[List[str]] = None


class ChatResponse(BaseModel):
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]
    

@sgl.function
def chat_function(s, prompt: str, max_tokens: int, temperature: float, top_p: float, stop: List[str] = None):
    """SGLang function for chat generation"""
    s += prompt
    s += gen("response", 
            max_tokens=max_tokens, 
            temperature=temperature, 
            top_p=top_p,
            stop=stop or ["<|im_end|>"])
