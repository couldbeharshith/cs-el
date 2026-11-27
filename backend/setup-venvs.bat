@echo off
uv run --script "%~dp0serve.py" --setup-only %*
exit /b %ERRORLEVEL%
