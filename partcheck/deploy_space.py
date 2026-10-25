"""Publishes PartCheck as a free public Hugging Face Space (Docker + Streamlit, CPU, 16 GB RAM) from this folder.
Use it if Streamlit Community Cloud runs out of memory. Your token stays on your computer.

    python deploy_space.py

Needs a Hugging Face WRITE token (huggingface.co/settings/tokens). Result: https://huggingface.co/spaces/<your-name>/partcheck
The trained models are NOT uploaded: the Space downloads them from the model repo on its first start (MODEL_REPO).
Note: free Spaces go to sleep when nobody visits for a while; the first visit afterwards wakes them (about a minute).
"""
import getpass
import os
import shutil
import tempfile

from huggingface_hub import HfApi

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ["streamlit_app.py", "partcheck_runtime.py", "triage_ui.py", "triage_engine.py", "triage_insights.py", "triage_image.py", "triage_core.py",
         "auto_data.py", "examples.json", "demo_auto_reviews.csv", "requirements.txt", "Dockerfile", ".streamlit/config.toml"]
SPACE_README = """---
title: PartCheck
emoji: 🔧
colorFrom: yellow
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

PartCheck: customer review and photo triage for an online automotive parts retailer.
"""

token = getpass.getpass("Paste your Hugging Face WRITE token (hidden), then press Enter: ").strip()
api = HfApi(token=token)
user = api.whoami()["name"]
repo = f"{user}/partcheck"
api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True)       # no hardware argument: the free CPU Space
tmp = tempfile.mkdtemp()
for name in FILES:
    src = os.path.join(HERE, name)
    if os.path.exists(src):
        os.makedirs(os.path.dirname(os.path.join(tmp, name)), exist_ok=True)
        shutil.copy(src, os.path.join(tmp, name))
    else:
        print("MISSING (skipped)", name)
with open(os.path.join(tmp, "README.md"), "w", encoding="utf-8") as f:
    f.write(SPACE_README)
api.upload_folder(folder_path=tmp, repo_id=repo, repo_type="space", commit_message="Deploy PartCheck (Streamlit)")
print()
print("Done. Your public link (the first build takes 5 to 10 minutes):")
print(f"   https://huggingface.co/spaces/{repo}")
print("Optional: in the Space's Settings > Variables and secrets, add APP_PASSWORD to protect the app.")
