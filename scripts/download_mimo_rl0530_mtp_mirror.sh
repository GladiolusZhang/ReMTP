#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Download MiMo-7B-RL-0530 and MiMo-7B-MTPs through https://hf-mirror.com.
Proxy variables are unset only inside the downloader process.

Usage:
  ./scripts/download_mimo_rl0530_mtp_mirror.sh

Interrupted downloads are resumed. Runtime uses local files and does not
connect to either Hugging Face or hf-mirror.
EOF
  exit 0
fi

# These exports affect only this script. The caller's proxy variables and
# shell configuration remain unchanged after the process exits.
export CLEAR_DOWNLOAD_PROXY=1
export HF_DOWNLOAD_ENDPOINT="${HF_DOWNLOAD_ENDPOINT:-https://hf-mirror.com}"
export REMTP_DIRECT_HF_DOWNLOAD=1

exec "$PROJECT_DIR/scripts/download_mimo_rl0530_mtp.sh" "$@"
