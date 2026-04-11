#!/usr/bin/env bash
cd "$(dirname "$0")"
[ -d venv ] || { echo "Creating venv..."; python3 -m venv venv; }
source venv/bin/activate
echo "Installing/updating dependencies..."
pip install -r requirements.txt -q
echo ""
echo "Starting Image analyzer (http://127.0.0.1:7860) ..."
python3 app.py
echo ""
read -p "Нажмите Enter для выхода..."
