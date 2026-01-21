@echo off
REM Setup Virtual Environments for All Backend Servers

echo Setting up virtual environments for all backend servers...
echo ============================================================

set BACKEND_PATH=%~dp0

REM Server 1: Main RAG API
echo.
echo Setting up Main RAG API venv...
if exist "%BACKEND_PATH%.venv_main" (
    echo   .venv_main already exists, skipping creation
) else (
    python -m venv "%BACKEND_PATH%.venv_main"
    echo   Created .venv_main
)
echo   Installing dependencies...
"%BACKEND_PATH%.venv_main\Scripts\python.exe" -m pip install --upgrade pip
"%BACKEND_PATH%.venv_main\Scripts\pip.exe" install -r "%BACKEND_PATH%requirements.txt"
echo   Main RAG API setup complete!

REM Server 2: Insurance RAG API
echo.
echo Setting up Insurance RAG API venv...
if exist "%BACKEND_PATH%rag_server\.venv_rag" (
    echo   .venv_rag already exists, skipping creation
) else (
    python -m venv "%BACKEND_PATH%rag_server\.venv_rag"
    echo   Created .venv_rag
)
echo   Installing dependencies...
"%BACKEND_PATH%rag_server\.venv_rag\Scripts\python.exe" -m pip install --upgrade pip
"%BACKEND_PATH%rag_server\.venv_rag\Scripts\pip.exe" install -r "%BACKEND_PATH%rag_server\requirements.txt"
echo   Insurance RAG API setup complete!

REM Server 3: Screener API
echo.
echo Setting up Screener API venv...
if exist "%BACKEND_PATH%screener_server\.venv_screener" (
    echo   .venv_screener already exists, skipping creation
) else (
    python -m venv "%BACKEND_PATH%screener_server\.venv_screener"
    echo   Created .venv_screener
)
echo   Installing dependencies...
"%BACKEND_PATH%screener_server\.venv_screener\Scripts\python.exe" -m pip install --upgrade pip
"%BACKEND_PATH%screener_server\.venv_screener\Scripts\pip.exe" install -r "%BACKEND_PATH%screener_server\requirements.txt"
echo   Screener API setup complete!

REM Server 4 & 5: MCP Servers
echo.
echo MCP Servers will use .venv_main

echo.
echo ============================================================
echo All virtual environments setup complete!
echo.
echo Virtual environments created:
echo   - .venv_main (Main RAG API + MCP Servers)
echo   - rag_server\.venv_rag (Insurance RAG API)
echo   - screener_server\.venv_screener (Screener API)
echo.
echo You can now run: start-all-servers.bat
echo.
pause
