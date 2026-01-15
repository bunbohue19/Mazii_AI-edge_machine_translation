"""
SGLang server for LLM. Serves the post-trained model
"""

import os
import yaml
import logging
import uvicorn
import sglang as sgl
from fastapi import FastAPI, HTTPException
from typing import Optional, List
from base_model.chat import ChatMessage
from base_model.translation import TranslationRequest, TranslationResponse, translation_function

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configurations
with open("config/config.yml") as f:
    config = yaml.safe_load(f)["VERSION"][os.getenv("VERSION")]

class SGLangServer:
    def __init__(self, model_path: str, adapter_path: Optional[str] = None, host: str = "0.0.0.0", port: int = 8888):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.system_prompt = config["PROMPT"]["SYSTEM"]
        self.user_prompt = config["PROMPT"]["USER"]
        self.host = host
        self.port = port
        self.app = FastAPI(title="SGLang Server with Adapter", version="1.0.0")
        self._setup_routes()
        self._validate_paths()
        print(self.user_prompt)
        
    def _validate_paths(self):
        """Validate model and adapter paths"""
        if not os.path.exists(self.model_path):
            raise ValueError(f"Model path does not exist: {self.model_path}")
        
        if self.adapter_path:
            if not os.path.exists(self.adapter_path):
                raise ValueError(f"Adapter path does not exist: {self.adapter_path}")
            
            # Check if adapter_model.safetensors exists
            adapter_file = os.path.join(self.adapter_path, "adapter_model.safetensors")
            if not os.path.exists(adapter_file):
                raise ValueError(f"adapter_model.safetensors not found in: {self.adapter_path}")
            
            logger.info(f"Found adapter weights: {adapter_file}")
        
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.on_event("startup")
        async def startup():
            """Initialize SGLang runtime on startup"""
            try:
                runtime_kwargs = {
                    "model_path": self.model_path,
                    "tokenizer_path": self.model_path,
                    "tp_size": 1,  # Adjust based on GPU setup
                    "mem_fraction_static": 0.80,  # Adjust based on available VRAM
                    "context_length": 8192,  # Adjust based on model capabilities
                }
                
                # Add adapter configuration if provided
                if self.adapter_path:
                    runtime_kwargs["lora_paths"] = [self.adapter_path]
                    logger.info(f"Loading base model: {self.model_path}")
                    logger.info(f"Loading LoRA adapter: {self.adapter_path}")
                else:
                    logger.info(f"Loading base model only: {self.model_path}")
                
                # Initialize SGLang runtime with the model and optional adapter
                sgl.set_default_backend(sgl.Runtime(**runtime_kwargs))
                
                if self.adapter_path:
                    logger.info("SGLang runtime initialized with base model + LoRA adapter")
                else:
                    logger.info("SGLang runtime initialized with base model only")
                    
            except Exception as e:
                logger.error(f"Failed to initialize SGLang runtime: {e}")
                logger.error("Make sure you have the correct SGLang version that supports LoRA adapters")
                raise
        
        @self.app.post("/v1/translate", response_model=TranslationResponse)
        async def translate(request: TranslationRequest):
            """Translation endpoint"""
            try:
                # Create the user prompt with template
                user_prompt = self.user_prompt.format(
                    target_lang_code=request.target_language_code,
                    text=request.text
                )
                
                # Create messages with system and user prompts
                messages = [
                    ChatMessage(role="system", content=self.system_prompt),
                    ChatMessage(role="user", content=user_prompt)
                ]
                
                # Convert to prompt format
                prompt = self._messages_to_prompt(messages)
                
                # Generate translation
                response = await self._generate_translation_response(
                    prompt=prompt,
                    thinking_budget=request.thinking_budget,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    target_language=request.target_language_code
                )
                
                return response
            except Exception as e:
                logger.error(f"Error in translation: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    def _messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        """Convert chat messages to Qwen prompt format"""
        prompt_parts = []
        for message in messages:
            if message.role == "system":
                prompt_parts.append(f"<|im_start|>system\n{message.content}<|im_end|>")
            elif message.role == "user":
                prompt_parts.append(f"<|im_start|>user\n{message.content}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")        # Add assistant start token for generation
        return "\n".join(prompt_parts)


    async def _generate_translation_response(self, prompt: str, thinking_budget: int, max_tokens: int, temperature: float, target_language: str) -> TranslationResponse:
        """Generate translation response using SGLang"""
        try:
            # Run the SGLang translation function
            state = translation_function.run(
                prompt=prompt,
                thinking_budget=thinking_budget,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            translated_text = state["translation"].strip()
            
            # Count tokens (approximate)
            prompt_tokens = len(prompt.split())
            completion_tokens = len(translated_text.split())
            total_tokens = prompt_tokens + completion_tokens
            
            return TranslationResponse(
                translated_text=translated_text,
                target_language=target_language,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
            )
        except Exception as e:
            logger.error(f"Error generating translation response: {e}")
            raise

    def run(self):
        """Start the server"""
        logger.info(f"Starting SGLang server on {self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port)