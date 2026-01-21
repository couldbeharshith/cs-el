@echo off
REM Stop All Backend Servers
REM This batch file stops all backend servers by killing processes on specific ports

echo Stopping all backend servers...
echo ============================================================

REM Stop processes on port 7860 (Insurance RAG API)
echo Stopping Insurance RAG API (Port 7860)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :7860 ^| findstr LISTENING') do taskkill /F /PID %%a 2>nul

REM Stop processes on port 8000 (Main RAG API)
echo Stopping Main RAG API (Port 8000)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do taskkill /F /PID %%a 2>nul

REM Stop processes on port 8001 (RAG MCP Server)
echo Stopping RAG MCP Server (Port 8001)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8001 ^| findstr LISTENING') do taskkill /F /PID %%a 2>nul

REM Stop processes on port 8002 (Computer MCP Server)
echo Stopping Computer MCP Server (Port 8002)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8002 ^| findstr LISTENING') do taskkill /F /PID %%a 2>nul

REM Stop processes on port 8080 (Screener API)
echo Stopping Screener API (Port 8080)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8080 ^| findstr LISTENING') do taskkill /F /PID %%a 2>nul

echo.
echo ============================================================
echo All backend servers stopped!
echo.
pause
