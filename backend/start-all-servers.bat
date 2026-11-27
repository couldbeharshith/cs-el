@echo off
uv run --script "%~dp0serve.py" %*
exit /b %ERRORLEVEL%
