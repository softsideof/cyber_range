"""Re-deploy all project files to HuggingFace Space."""

import os
import sys

# Force UTF-8
os.environ["PYTHONUTF8"] = "1"

from huggingface_hub import HfApi

REPO_ID = "keshav-005/cyber_range"
TOKEN = os.getenv("HF_TOKEN")

if not TOKEN:
    print("Set HF_TOKEN environment variable first!")
    sys.exit(1)

api = HfApi(token=TOKEN)

# Upload the entire project folder
api.upload_folder(
    folder_path=".",
    repo_id=REPO_ID,
    repo_type="space",
    ignore_patterns=[
        ".git/*",
        ".git/**",
        "__pycache__/*",
        "__pycache__/**",
        "**/__pycache__/*",
        "**/__pycache__/**",
        "*.pyc",
        ".pytest_cache/*",
        ".pytest_cache/**",
        "validation_source.txt",
        "validation_results.log",
        "deploy_to_hf.py",
        ".venv/*",
        ".venv/**",
        "README.md",
    ],
)

# Upload the specific HuggingFace config as README.md
api.upload_file(
    path_or_fileobj="HF_README.md",
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="space",
)

print("Upload complete!")
print(f"Space: https://huggingface.co/spaces/{REPO_ID}")
