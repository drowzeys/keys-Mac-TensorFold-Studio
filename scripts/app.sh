#!/bin/bash
# The TensorFold Studio web app: create wizard, production board, clip editor.
#   bash scripts/app.sh                 # http://127.0.0.1:7870
#   HOST=0.0.0.0 bash scripts/app.sh    # also reachable from other machines on your network (no login: trusted LANs only)
# Environment: PREFIX, QWEN_MODEL_DIR, H3_MODEL_DIR (as for the other scripts), HOST, PORT, STUDIO_HOME (default
# ~/TensorFoldStudio, where projects, uploads and renders are kept).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
export PATH="/opt/homebrew/bin:$PATH"
[ -x "$PREFIX/venv/bin/python" ] || { echo "run oneshot-setup.sh first: no environment at $PREFIX" >&2; exit 1; }
echo "TensorFold Studio on http://${HOST:-127.0.0.1}:${PORT:-7870}"
exec "$PREFIX/venv/bin/python" "$HERE/app/server.py"
