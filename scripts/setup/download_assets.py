#!/usr/bin/env python3
"""Repository wrapper for the installed somehand asset downloader.

Usage:
    python scripts/setup/download_assets.py
    python scripts/setup/download_assets.py --only mjcf mediapipe
    python scripts/setup/download_assets.py --source huggingface --repo-id <repo>
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from somehand.asset_download import main


if __name__ == "__main__":
    main()
