#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "=== Activating virtualenv ==="
source venv/bin/activate

echo ""
echo "=== Running ruff ==="
python -m ruff check src/ tests/ monitoring/ 2>&1 || true
echo "RUFF_EXIT:$?"

echo ""
echo "=== Running flake8 ==="
python -m flake8 src/ tests/ monitoring/ --max-line-length=100 --extend-ignore=E203,W503 2>&1 || true
echo "FLAKE8_EXIT:$?"
