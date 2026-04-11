#!/usr/bin/env python3
"""
Download DeepFashion Attribute Prediction dataset (2026).

Official page: https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html
Google Drive folder (no password): https://drive.google.com/drive/folders/0B7EVK8r0v71pQ2FuZ0k0QnhBQnc

Usage:
  pip install gdown
  python scripts/download_deepfashion.py [--out data/deepfashion]

Downloads the folder to --out; you need at least "img" (images) and "Anno" (list_attr_cloth.txt, list_attr_img.txt).
"""

import argparse
import sys
from pathlib import Path

GOOGLE_DRIVE_FOLDER_ID = "0B7EVK8r0v71pQ2FuZ0k0QnhBQnc"


def main():
    p = argparse.ArgumentParser(description="Download DeepFashion Attribute Prediction dataset")
    p.add_argument("--out", default="data/deepfashion", help="Output directory")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    try:
        import gdown
    except ImportError:
        print("Install gdown: pip install gdown", file=sys.stderr)
        sys.exit(1)

    url = f"https://drive.google.com/drive/folders/{GOOGLE_DRIVE_FOLDER_ID}"
    print(f"Downloading DeepFashion to {out.resolve()} ...")
    gdown.download_folder(url, output=str(out), quiet=False, use_cookies=False)
    print("Done. Ensure Anno/list_attr_cloth.txt and Anno/list_attr_img.txt exist.")
    print("Images are typically in img/ or similar subfolder.")


if __name__ == "__main__":
    main()
