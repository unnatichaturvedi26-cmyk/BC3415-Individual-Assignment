"""Publishes PartCheck as a permanent public web page (a free Hugging Face Space, CPU) from this folder.
Run it yourself: your token stays on your computer.

    python deploy_space.py

Needs a Hugging Face WRITE token (huggingface.co/settings/tokens). Result: https://huggingface.co/spaces/<your-name>/partcheck
The trained models are NOT uploaded here: the Space downloads them from your model repo (unnati002/review-triage-files) on its first start.
"""
import getpass
import os
import tempfile

from huggingface_hub import HfApi

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = ["app.py", "triage_ui.py", "triage_engine.py", "triage_insights.py", "triage_image.py", "triage_core.py", "auto_data.py", "examples.json",
        "demo_auto_reviews.csv", "requirements.txt"]
SPACE_README = """---
title: PartCheck
emoji: 🔧
colorFrom: yellow
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

PartCheck: customer review and photo triage for an online automotive parts retailer.
"""

token = getpass.getpass("Paste your Hugging Face WRITE token (hidden), then press Enter: ").strip()
api = HfApi(token=token)
user = api.whoami()["name"]
repo = f"{user}/partcheck"
api.create_repo(repo, repo_type="space", space_sdk="gradio", exist_ok=True)       # no hardware argument: the free CPU Space
tmp = tempfile.mkdtemp()
with open(os.path.join(tmp, "README.md"), "w", encoding="utf-8") as f:
    f.write(SPACE_README)
api.upload_file(path_or_fileobj=os.path.join(tmp, "README.md"), path_in_repo="README.md", repo_id=repo, repo_type="space")
for name in CODE:
    path = os.path.join(HERE, name)
    if os.path.exists(path):
        api.upload_file(path_or_fileobj=path, path_in_repo=name, repo_id=repo, repo_type="space")
        print("uploaded", name)
    else:
        print("MISSING (skipped)", name)
print()
print("Done. Your public link (it takes 3 to 6 minutes to build the first time):")
print(f"   https://huggingface.co/spaces/{repo}")
