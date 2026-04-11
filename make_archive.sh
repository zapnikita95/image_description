#!/usr/bin/env bash
# Create a zip archive of the project for transferring to another computer.
# Excludes: venv, __pycache__, .pyc, .env, existing zip files.
# Run from the project directory: ./make_archive.sh

cd "$(dirname "$0")"
NAME="feed_image_attributes"
ZIP="${NAME}.zip"
# Optional: include date in filename
# ZIP="${NAME}_$(date +%Y%m%d).zip"

if [ -f "$ZIP" ]; then
  rm -f "$ZIP"
fi
zip -r "$ZIP" . \
  -x "venv/*" \
  -x "__pycache__/*" \
  -x "*.pyc" \
  -x ".env" \
  -x "*.zip" \
  -x ".DS_Store"

echo "Created: $(pwd)/$ZIP"
echo "Transfer this file to another computer, unzip, then follow README.md (install Ollama, pull model, pip install -r requirements.txt, run)."
