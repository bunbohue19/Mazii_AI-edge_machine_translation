import argparse
from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser()
parser.add_argument("--repo_id", type=str)
parser.add_argument("--filename", type=str)
args = parser.parse_args()

hf_hub_download(repo_id=args.repo_id, filename=args.filename)