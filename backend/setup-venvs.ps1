# Setup Virtual Environments for All Backend Servers
# This script creates separate venvs and installs dependencies for each server

Write-Host "Setting up virtual environments for all backend servers..." -ForegroundColor Green
Write-Host "=" * 60

$backendPath = $PSScriptRoot

# Server 1: Main RAG API
Write-Host "`nSetting up Main RAG API venv..." -ForegroundColor Cyan
if (Test-Path "$backendPath\.venv_main") {
    Write-Host "  .venv_main already exists, skipping creation" -ForegroundColor Yellow
} else {
    python -m venv "$backendPath\.venv_main"
    Write-Host "  Created .venv_main" -ForegroundColor Green
}
Write-Host "  Installing dependencies..." -ForegroundColor Gray
& "$backendPath\.venv_main\Scripts\python.exe" -m pip install --upgrade pip
& "$backendPath\.venv_main\Scripts\pip.exe" install -r "$backendPath\requirements.txt"
Write-Host "  Main RAG API setup complete!" -ForegroundColor Green

# Server 2: Insurance RAG API
Write-Host "`nSetting up Insurance RAG API venv..." -ForegroundColor Cyan
if (Test-Path "$backendPath\rag_server\.venv_rag") {
    Write-Host "  .venv_rag already exists, skipping creation" -ForegroundColor Yellow
} else {
    python -m venv "$backendPath\rag_server\.venv_rag"
    Write-Host "  Created .venv_rag" -ForegroundColor Green
}
Write-Host "  Installing dependencies..." -ForegroundColor Gray
& "$backendPath\rag_server\.venv_rag\Scripts\python.exe" -m pip install --upgrade pip
& "$backendPath\rag_server\.venv_rag\Scripts\pip.exe" install -r "$backendPath\rag_server\requirements.txt"
Write-Host "  Insurance RAG API setup complete!" -ForegroundColor Green

# Server 3: Screener API
Write-Host "`nSetting up Screener API venv..." -ForegroundColor Cyan
if (Test-Path "$backendPath\screener_server\.venv_screener") {
    Write-Host "  .venv_screener already exists, skipping creation" -ForegroundColor Yellow
} else {
    python -m venv "$backendPath\screener_server\.venv_screener"
    Write-Host "  Created .venv_screener" -ForegroundColor Green
}
Write-Host "  Installing dependencies..." -ForegroundColor Gray
& "$backendPath\screener_server\.venv_screener\Scripts\python.exe" -m pip install --upgrade pip
& "$backendPath\screener_server\.venv_screener\Scripts\pip.exe" install -r "$backendPath\screener_server\requirements.txt"
Write-Host "  Screener API setup complete!" -ForegroundColor Green

# Server 4 & 5: MCP Servers (share main venv)
Write-Host "`nMCP Servers will use .venv_main" -ForegroundColor Cyan

Write-Host ""
Write-Host "=" * 60
Write-Host "All virtual environments setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Virtual environments created:" -ForegroundColor Yellow
Write-Host "  - .venv_main (Main RAG API + MCP Servers)" -ForegroundColor White
Write-Host "  - rag_server\.venv_rag (Insurance RAG API)" -ForegroundColor White
Write-Host "  - screener_server\.venv_screener (Screener API)" -ForegroundColor White
Write-Host ""
Write-Host "You can now run: .\start-all-servers.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
