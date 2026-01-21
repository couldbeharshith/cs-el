# Start All Backend Servers
# This script starts all required backend servers in separate PowerShell windows
# Each server uses its own virtual environment

Write-Host "Starting all backend servers..." -ForegroundColor Green
Write-Host "=" * 60

$backendPath = $PSScriptRoot

# Check if venvs exist
if (-not (Test-Path "$backendPath\.venv_main")) {
    Write-Host "ERROR: Virtual environments not found!" -ForegroundColor Red
    Write-Host "Please run .\setup-venvs.ps1 first" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Press any key to exit..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# Server 1: Insurance RAG API (Port 7860)
Write-Host "Starting Insurance RAG API Server (Port 7860)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$backendPath\rag_server'; .\.venv_rag\Scripts\python.exe app.py"

# Wait a bit before starting next server
Start-Sleep -Seconds 3

# Server 2: Main RAG API (Port 8000)
Write-Host "Starting Main RAG API Server (Port 8000)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$backendPath'; .\.venv_main\Scripts\python.exe main.py"

# Wait a bit before starting next server
Start-Sleep -Seconds 2

# Server 3: RAG MCP Server (Port 8001)
Write-Host "Starting RAG MCP Server (Port 8001)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$backendPath'; .\.venv_main\Scripts\python.exe run_mcp.py"

# Wait a bit before starting next server
Start-Sleep -Seconds 2

# Server 4: Computer MCP Server (Port 8002)
Write-Host "Starting Computer MCP Server (Port 8002)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$backendPath'; .\.venv_main\Scripts\python.exe run_computer_mcp.py"

# Wait a bit before starting next server
Start-Sleep -Seconds 2

# Server 5: Screener API Server (Port 8080)
Write-Host "Starting Screener API Server (Port 8080)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$backendPath\screener_server'; .\.venv_screener\Scripts\python.exe server.py"

Write-Host ""
Write-Host "=" * 60
Write-Host "All backend servers started!" -ForegroundColor Green
Write-Host ""
Write-Host "Servers running on:" -ForegroundColor Yellow
Write-Host "  - Insurance RAG API:   http://localhost:7860" -ForegroundColor White
Write-Host "  - Main RAG API:        http://localhost:8000" -ForegroundColor White
Write-Host "  - RAG MCP Server:      http://localhost:8001/mcp" -ForegroundColor White
Write-Host "  - Computer MCP Server: http://localhost:8002/mcp" -ForegroundColor White
Write-Host "  - Screener API:        http://localhost:8080" -ForegroundColor White
Write-Host ""
Write-Host "To start the frontend, run:" -ForegroundColor Yellow
Write-Host "  cd frontend" -ForegroundColor White
Write-Host "  npm run dev" -ForegroundColor White
Write-Host ""
Write-Host "Press any key to close this window (servers will keep running)..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
