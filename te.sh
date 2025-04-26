export PATH=/usr/bin:$PATH
set -x

rm -rf .venv
uv sync
uv sync --extra mcore
uv run te.py
