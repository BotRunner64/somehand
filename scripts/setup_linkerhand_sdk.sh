#!/usr/bin/env bash
# Ensure LinkerHand Python SDK submodule is available for the real backend.
# Also installs the SDK's Python dependencies from requirements.txt.
# Usage: bash scripts/setup_linkerhand_sdk.sh [sdk_dir]
#
# Default sdk_dir: ./third_party/linkerhand-python-sdk

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK_DIR="${1:-$REPO_ROOT/third_party/linkerhand-python-sdk}"
SDK_DIR="$(mkdir -p "$(dirname "$SDK_DIR")" && cd "$(dirname "$SDK_DIR")" && pwd)/$(basename "$SDK_DIR")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SKIP_PIP_INSTALL="${SKIP_PIP_INSTALL:-0}"

echo "==> Using LinkerHand SDK at: $SDK_DIR"

if [ ! -f "$SDK_DIR/LinkerHand/linker_hand_api.py" ]; then
    echo "==> LinkerHand SDK submodule missing, trying to initialize it..."
    git -C "$REPO_ROOT" submodule update --init --recursive -- "third_party/linkerhand-python-sdk" || true
fi

if [ ! -f "$SDK_DIR/LinkerHand/linker_hand_api.py" ]; then
    echo "==> ERROR: LinkerHand SDK not found at: $SDK_DIR"
    echo "==> Please run: git submodule update --init --recursive"
    exit 1
fi

if [ ! -f "$SDK_DIR/LinkerHand/utils/mapping.py" ]; then
    echo "==> ERROR: LinkerHand mapping module missing at: $SDK_DIR/LinkerHand/utils/mapping.py"
    exit 1
fi

if [ -f "$SDK_DIR/requirements.txt" ]; then
    if [ "$SKIP_PIP_INSTALL" = "1" ]; then
        echo "==> Skipping LinkerHand SDK Python dependency installation (SKIP_PIP_INSTALL=1)."
    else
        if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
            echo "==> ERROR: Python interpreter not found: $PYTHON_BIN"
            echo "==> Tip: set PYTHON_BIN=/path/to/python and rerun this script."
            exit 1
        fi
        echo "==> Installing LinkerHand SDK Python dependencies via: $PYTHON_BIN -m pip install -r $SDK_DIR/requirements.txt"
        "$PYTHON_BIN" -m pip install -r "$SDK_DIR/requirements.txt"
    fi
else
    echo "==> WARNING: requirements.txt not found at: $SDK_DIR/requirements.txt"
fi

echo "==> LinkerHand SDK is ready."
echo "==> Default somehand sdk_root now points here:"
echo "    $SDK_DIR"
