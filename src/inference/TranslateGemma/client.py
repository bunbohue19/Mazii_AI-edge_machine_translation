from typing import Dict, List

class Client:
    """Simple client for testing the server"""
    
    def __init__(self, base_url: str = "http://localhost:8888"):
        self.base_url = base_url
    
    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs):
        """Send chat completion request"""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                json={"messages": messages, **kwargs}
            ) as response:
                return await response.json()
    
    async def completion(self, prompt: str, **kwargs):
        """Send completion request"""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/completions", 
                json={"prompt": prompt, **kwargs}
            ) as response:
                return await response.json()
    
    async def translate(self, text: str, target_lang_code: str, **kwargs):
        """Send translation request"""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/translate",
                json={
                    "text": text,
                    "target_lang_code": target_lang_code,
                    **kwargs
                }
            ) as response:
                return await response.json()