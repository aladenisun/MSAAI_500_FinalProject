#!/usr/bin/env bash
set -euo pipefail

# Usage: run this from the folder where requirements.txt lives.
# Example (creates ./.venv here and registers a kernel):
#   chmod +x setup_venv.sh
#   ./setup_venv.sh ./.venv "my-env" "Python (My Env)"
#
# Defaults:
#   ENV_PATH -> ./.venv
#   KERNEL_NAME -> venv
#   KERNEL_DISPLAY -> Python (venv)

ENV_PATH="${1:-.venv}"
KERNEL_NAME="${2:-venv}"
KERNEL_DISPLAY="${3:-Python (venv)}"

python3 -m venv "$ENV_PATH"
source "$ENV_PATH/bin/activate"

python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt

# Register a Jupyter kernel for this venv (optional but handy)
python -m ipykernel install --user --name="$KERNEL_NAME" --display-name="$KERNEL_DISPLAY"

echo ""
echo "Done! Virtual environment created at: $ENV_PATH"
echo "To start using it now:"
echo "  source \"$ENV_PATH/bin/activate\""
echo "Then run Jupyter with:"
echo "  jupyter-lab"
