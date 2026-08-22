#!/usr/bin/env bash
# Linux / macOS Setup Script
set -e

echo "Setting up Industrial Defect Detection Environment..."

# 1. Check Python
python3 --version

# 2. Create Virtual Environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in .venv..."
    python3 -m venv .venv
fi

# 3. Activate Virtual Environment
echo "Activating virtual environment..."
source .venv/bin/activate

# 4. Install Dependencies
echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt

echo -e "\nSetup complete! You can now run:"
echo "  python -m src.data.download_dataset"
echo "  python -m src.data.prepare_dataset"
echo "  python -m src.training.train"
echo "  uvicorn api.main:app --reload"
