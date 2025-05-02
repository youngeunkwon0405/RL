export PATH=/usr/bin:$PATH
set -xeu

rm -rf .venv
uv sync
uv sync --extra mcore
uv run te.py
