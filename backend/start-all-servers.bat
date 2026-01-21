@echo off
REM Start All Backend Servers
REM This batch file starts all required backend servers in separate windows
REM Each server uses its own virtual environment

echo Starting all backend servers...
echo ============================================================

REM Get the directory where this script is located
set BACKEND_PATH=%~dp0

REM Check if venvs exist
if not exist "%BACKEND_PATH%.venv_main" (
    echo ERROR: Virtual environments not found!
    echo Please run setup-venvs.bat first
    echo.
    pause
    exit /b 1
)

REM Server 1: Insurance RAG API (Port 7860)
echo Starting Insurance RAG API Server (Port 7860)...
start "Insurance RAG API (7860)" cmd /k "cd /d %BACKEND_PATH%rag_server && .venv_rag\Scripts\python.exe app.py"

REM Wait a bit before starting next server
timeout /t 3 /nobreak >nul

REM Server 2: Main RAG API (Port 8000)
echo Starting Main RAG API Server (Port 8000)...
start "Main RAG API (8000)" cmd /k "cd /d %BACKEND_PATH% && .venv_main\Scripts\python.exe main.py"

REM Wait a bit before starting next server
timeout /t 2 /nobreak >nul

REM Server 3: RAG MCP Server (Port 8001)
echo Starting RAG MCP Server (Port 8001)...
start "RAG MCP Server (8001)" cmd /k "cd /d %BACKEND_PATH% && .venv_main\Scripts\python.exe run_mcp.py"

REM Wait a bit before starting next server
timeout /t 2 /nobreak >nul

REM Server 4: Computer MCP Server (Port 8002)
echo Starting Computer MCP Server (Port 8002)...
start "Computer MCP Server (8002)" cmd /k "cd /d %BACKEND_PATH% && .venv_main\Scripts\python.exe run_computer_mcp.py"

REM Wait a bit before starting next server
timeout /t 2 /nobreak >nul

REM Server 5: Screener API Server (Port 8080)
echo Starting Screener API Server (Port 8080)...
start "Screener API (8080)" cmd /k "cd /d %BACKEND_PATH%screener_server && .venv_screener\Scripts\python.exe server.py"

echo.
echo ============================================================
echo All backend servers started!
echo.
echo Servers running on:
echo   - Insurance RAG API:   http://localhost:7860
echo   - Main RAG API:        http://localhost:8000
echo   - RAG MCP Server:      http://localhost:8001/mcp
echo   - Computer MCP Server: http://localhost:8002/mcp
echo   - Screener API:        http://localhost:8080
echo.
echo To start the frontend, run:
echo   cd frontend
echo   npm run dev
echo.
pause
