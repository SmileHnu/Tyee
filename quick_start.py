#!/usr/bin/env python3
# Quick start script for Tyee (cross-platform)
# This script prepares data and runs the MIT-BIH experiment.
# Comments are in English; prompts are in Chinese for clarity.

import os
import sys
import subprocess
from pathlib import Path
import urllib.request

# 强制 UTF-8 输出，避免 GBK 编码错误（Windows）
os.environ.setdefault("PYTHONUTF8", "1")
try:
    # Python 3.7+ 支持 reconfigure
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "original"
DATA_URL = "https://physionet.org/files/mitdb/1.0.0/"
CONFIG_PATH = PROJECT_ROOT / "tyee" / "config" / "mit_bih.yaml"


def ensure_dirs():
    """Create needed directories."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Data directory created: {DATA_DIR}")


def download_using_urllib(url, dest_dir):
    """
    Download a single file using urllib. Note: urllib does not mirror directories.
    For full recursive mirroring prefer wget on Linux/WSL or use the shell script.
    """
    fname = url.rstrip('/').split('/')[-1] or "index.html"
    dest = Path(dest_dir) / fname
    print(f"Downloading {url} -> {dest}")
    try:
        urllib.request.urlretrieve(url, dest)
        print("Download finished (single file).")
    except Exception as e:
        print("Download failed:", e)
        return False
    return True


def run_training():
    """Run the training launcher using the current Python interpreter."""
    py = os.environ.get("PYTHON", sys.executable)
    if not py:
        print("错误：找不到 python。请确保已安装 Python 或激活 Conda 环境。")
        sys.exit(1)
    cmd = [py, str(PROJECT_ROOT / "train.py"), "-c", str(CONFIG_PATH)]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    print("=== Tyee Quick Start ===")
    ensure_dirs()
    # skip download if directory not empty
    if any(DATA_DIR.iterdir()):
        print("检测到已有数据，跳过下载步骤")
    else:
        ok = download_using_urllib(DATA_URL, DATA_DIR)
        if not ok:
            print("建议手动下载或在 Linux 上使用 wget 以完整镜像文件。")
    run_training()
    print("=== Done ===")


if __name__ == '__main__':
    main()
