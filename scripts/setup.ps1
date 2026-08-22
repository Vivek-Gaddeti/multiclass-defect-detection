# Windows PowerShell Setup Script
Write-Host "Setting up Industrial Defect Detection Environment..." -ForegroundColor Cyan

# 1. Check Python
$pythonVersion = python --version
Write-Host "Detected: $pythonVersion" -ForegroundColor Green

# 2. Create Virtual Environment if not present
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment in .venv..." -ForegroundColor Yellow
    python -m venv .venv
}

# 3. Activate Virtual Environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .venv\Scripts\Activate.ps1

# 4. Install Dependencies
Write-Host "Installing requirements..." -ForegroundColor Yellow
pip install -r requirements.txt
pip install -r requirements-dev.txt

Write-Host "`nSetup complete! You can now run:" -ForegroundColor Green
Write-Host "  python -m src.data.download_dataset"
Write-Host "  python -m src.data.prepare_dataset"
Write-Host "  python -m src.training.train"
Write-Host "  uvicorn api.main:app --reload"
