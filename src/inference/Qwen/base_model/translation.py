import sglang as sgl
from typing import Optional, Dict
from pydantic import BaseModel
from sglang import gen


class TranslationRequest(BaseModel):
    text: str
    target_language_code: str
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.20  # Lower temperature for translation tasks
    thinking_budget: Optional[int] = 128

class TranslationResponse(BaseModel):
    translated_text: str
    target_language: str
    usage: Dict[str, int]
    

@sgl.function
def translation_function(s, prompt: str, thinking_budget: int = 128, max_tokens: int = 1024, temperature: float = 0.20):
    """SGLang function optimized for translation tasks"""
    # # First stage: generate thinking content
    # s += prompt
    # s += sgl.gen("thinking", max_tokens=thinking_budget, stop=["</think>"])
    
    # # Check if thinking ended naturally or budget exceeded
    # thinking_text = s["thinking"]
    # if "</think>" not in thinking_text:
    #     # Force close thinking
    #     s += "\n\nConsidering the limited time, I'll provide the translation now.\n</think>\n\n"
    
    # # Second stage: generate actual translation
    # s += sgl.gen("translation", max_tokens=max_tokens, temperature=temperature, stop=["<|im_end|>"])
    
    s += prompt
    s += "<think>\n\n</think>\n\n"
    s += gen("translation", 
            max_tokens=max_tokens, 
            temperature=temperature,  
            top_p=0.95,
            top_k=20,
            stop=["<|im_end|>"]
        )