#!/bin/bash
cd "$(dirname "$0")" || exit 1
if ! command -v python3 >/dev/null 2>&1; then
  echo "Install Python 3.11 or 3.12 from python.org first."
  read -r -p "Press Enter to close..."; exit 1
fi
if [ ! -d .venv ]; then python3 -m venv .venv || exit 1; fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt || { echo "Installation failed; check above."; read -r -p "Press Enter to close..."; exit 1; }
python -m streamlit run streamlit_app.py --server.address localhost --server.port 8501
read -r -p "Press Enter to close..."
