import argparse
import logging
from server import SGLangServer


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


if __name__ == "__main__":
    main()