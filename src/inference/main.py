import os
import argparse
import logging
from server import SGLangServer
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_env_file(env_path: Path) -> None:
    """Populate os.environ entries from a simple KEY=VALUE .env file."""
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key:
            os.environ.setdefault(key.strip(), value.strip())

def main():
    parser = argparse.ArgumentParser(description="SGLang Server with LoRA Adapter")
    parser.add_argument("--model-path", type=str, required=True,
                    help="Path to the base LLM")
    parser.add_argument("--adapter-path", type=str, default=None,
                    help="Path to the LoRA adapter directory (containing adapter_model.safetensors)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                    help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8888,
                    help="Port to bind the server to")
    parser.add_argument("--test", action="store_true",
                    help="Run test client instead of server")
    
    args = parser.parse_args()
    
    server = SGLangServer(  
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        host=args.host, 
        port=args.port 
    )
    
    server.run()

PROJECT_ROOT = os.getenv("PROJECT_ROOT")
load_env_file(PROJECT_ROOT / ".env")
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise RuntimeError("HF_TOKEN is not set in the environment or .env file.")
login(token=hf_token)

if __name__ == "__main__":
    main()