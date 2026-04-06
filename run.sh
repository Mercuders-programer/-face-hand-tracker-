#!/bin/bash
# WSL2에서 PoseTracker 실행 스크립트
# 사용법: bash run.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LD_LIBRARY_PATH="$SCRIPT_DIR/lib:$LD_LIBRARY_PATH"

/usr/bin/python3 "$SCRIPT_DIR/app.py" "$@"
