import sglang as sgl
from typing import Optional, Dict
from pydantic import BaseModel
from sglang import gen


class TranslationRequest(BaseModel):
    text: str
    target_language_code: str
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.10  # Lower temperature for translation tasks

class TranslationResponse(BaseModel):
    translated_text: str
    target_language: str
    usage: Dict[str, int]
    

@sgl.function
def translation_function(s, prompt: str, max_tokens: int = 1024, temperature: float = 0.10):
    """SGLang function optimized for translation tasks"""
    s += prompt
    s += gen("translation", 
            max_tokens=max_tokens, 
            temperature=temperature,  
            top_p=0.95,
            top_k=20,
            stop=["<|im_end|>"]
        )
