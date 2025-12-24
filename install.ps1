# Check if uv is installed
if (-not (Get-Command "uv" -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..."
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    # Refresh env vars for current session
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","User") + ";" + [System.Environment]::GetEnvironmentVariable("Path","Machine")
}

Write-Host "Creating virtual environment..."
uv venv .venv --python 3.10 --clear

Write-Host "Installing dependencies..."
# Activate venv for the installation context or use uv pip directly
uv pip install -r src/embedding_server_py/requirements.txt

# Use XPU
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
# Or else use normal torch
# uv pip install torch torchvision torchaudio

Write-Host "Setting up models (downloading and converting to ONNX)..."
uv run src/embedding_server_py/setup_models.py

Write-Host "Installation complete!"
Write-Host "To activate the environment, run: .\.venv\Scripts\activate"
Write-Host "To start the server, run: python src/embedding_server_py/main_es.py"
