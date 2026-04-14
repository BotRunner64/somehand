#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_ROOT="$ROOT_DIR/configs/retargeting"
RECORDINGS_DIR="$ROOT_DIR/recordings"
OUTPUT_ROOT="$RECORDINGS_DIR/full_renders"
LANDMARK_SCRIPT="$ROOT_DIR/scripts/render_landmark_video.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"
FORCE=0
RENDER_LANDMARKS=1

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --output-root PATH    Output root directory. Default: $OUTPUT_ROOT
  --recordings-dir PATH Recordings directory. Default: $RECORDINGS_DIR
  --configs-root PATH   Config root directory. Default: $CONFIG_ROOT
  --python BIN          Python executable. Default: $PYTHON_BIN
  --force               Re-render videos even if output already exists
  --skip-landmarks      Skip rendering landmarks-only comparison videos
  -h, --help            Show this help

This script renders all concrete configs under:
  - configs/retargeting/right/*.yaml
  - configs/retargeting/left/*.yaml
  - configs/retargeting/bihand/*.yaml

It uses these full-length recordings by default:
  - recordings/pico_right.pkl
  - recordings/pico_left.pkl
  - recordings/pico_bihand.pkl

Outputs are grouped under:
  - <output-root>/right/
  - <output-root>/left/
  - <output-root>/bihand/
  - <output-root>/landmarks/right/
  - <output-root>/landmarks/left/
  - <output-root>/landmarks/bihand/
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --recordings-dir)
            RECORDINGS_DIR="$2"
            shift 2
            ;;
        --configs-root)
            CONFIG_ROOT="$2"
            shift 2
            ;;
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --force)
            FORCE=1
            shift
            ;;
        --skip-landmarks)
            RENDER_LANDMARKS=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

RIGHT_RECORDING="$RECORDINGS_DIR/pico_right.pkl"
LEFT_RECORDING="$RECORDINGS_DIR/pico_left.pkl"
BIHAND_RECORDING="$RECORDINGS_DIR/pico_bihand.pkl"

for required_path in \
    "$CONFIG_ROOT/right" \
    "$CONFIG_ROOT/left" \
    "$CONFIG_ROOT/bihand" \
    "$LANDMARK_SCRIPT" \
    "$RIGHT_RECORDING" \
    "$LEFT_RECORDING" \
    "$BIHAND_RECORDING"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Missing required path: $required_path" >&2
        exit 1
    fi
done

mkdir -p \
    "$OUTPUT_ROOT/right" \
    "$OUTPUT_ROOT/left" \
    "$OUTPUT_ROOT/bihand" \
    "$OUTPUT_ROOT/landmarks/right" \
    "$OUTPUT_ROOT/landmarks/left" \
    "$OUTPUT_ROOT/landmarks/bihand"

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

rendered=()
skipped=()
failed=()

render_group() {
    local group="$1"
    local recording="$2"
    local hand="$3"
    local output_dir="$OUTPUT_ROOT/$group"
    local config_path

    for config_path in "$CONFIG_ROOT/$group"/*.yaml; do
        local config_name
        local output_path

        config_name="$(basename "${config_path%.yaml}")"
        output_path="$output_dir/${config_name}.mp4"

        if [[ -f "$output_path" && "$FORCE" -ne 1 ]]; then
            echo "[skip] $group/$config_name -> $output_path"
            skipped+=("$output_path")
            continue
        fi

        echo "[render] $group/$config_name"
        if "$PYTHON_BIN" -m somehand.cli dump-video \
            --config "$config_path" \
            --hand "$hand" \
            --recording "$recording" \
            --output "$output_path"; then
            rendered+=("$output_path")
        else
            failed+=("$config_path")
        fi
    done
}

render_group "right" "$RIGHT_RECORDING" "right"
render_group "left" "$LEFT_RECORDING" "left"
render_group "bihand" "$BIHAND_RECORDING" "both"

render_landmark_video() {
    local group="$1"
    local recording="$2"
    local mode="$3"
    local recording_stem
    local output_path

    recording_stem="$(basename "${recording%.pkl}")"
    output_path="$OUTPUT_ROOT/landmarks/$group/${recording_stem}_landmarks.mp4"

    if [[ -f "$output_path" && "$FORCE" -ne 1 ]]; then
        echo "[skip] landmarks/$group -> $output_path"
        skipped+=("$output_path")
        return
    fi

    echo "[render] landmarks/$group"
    if "$PYTHON_BIN" "$LANDMARK_SCRIPT" \
        --recording "$recording" \
        --output "$output_path" \
        --mode "$mode"; then
        rendered+=("$output_path")
    else
        failed+=("$recording -> $output_path")
    fi
}

if [[ "$RENDER_LANDMARKS" -eq 1 ]]; then
    render_landmark_video "right" "$RIGHT_RECORDING" "single"
    render_landmark_video "left" "$LEFT_RECORDING" "single"
    render_landmark_video "bihand" "$BIHAND_RECORDING" "bihand"
fi

echo
echo "Rendered: ${#rendered[@]}"
echo "Skipped: ${#skipped[@]}"
echo "Failed: ${#failed[@]}"

if [[ ${#failed[@]} -gt 0 ]]; then
    printf 'Failed configs:\n'
    printf '  %s\n' "${failed[@]}"
    exit 1
fi
