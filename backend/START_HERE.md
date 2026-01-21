# Backend Server Management

This directory contains scripts to easily start and stop all backend servers.

## First Time Setup

**IMPORTANT: Run this first to create virtual environments:**

### Option 1: Using PowerShell (Recommended)
```powershell
.\setup-venvs.ps1
```

### Option 2: Using Batch File
```cmd
setup-venvs.bat
```

This creates 3 separate virtual environments:
- `.venv_main` - Main RAG API + MCP Servers
- `rag_server\.venv_rag` - Insurance RAG API
- `screener_server\.venv_screener` - Screener API

## Starting Servers

### Option 1: Using PowerShell (Recommended)
```powershell
# Start all servers
.\start-all-servers.ps1

# Stop all servers
.\stop-all-servers.ps1
```

### Option 2: Using Batch File
```cmd
# Start all servers
start-all-servers.bat

# Stop all servers
stop-all-servers.bat
```

## What Gets Started

The script starts 5 backend servers in separate windows:

1. **Insurance RAG API** (Port 7860)
   - Advanced RAG pipeline with Pinecone, Gemini, and Groq
   - Supports PDF, DOCX, XLSX, PPTX, images, audio, video
   - URL: http://localhost:7860

2. **Main RAG API** (Port 8000)
   - Calls the Insurance RAG API for document processing
   - URL: http://localhost:8000

3. **RAG MCP Server** (Port 8001)
   - MCP server for RAG tools (calls Insurance RAG API)
   - URL: http://localhost:8001/mcp

4. **Computer MCP Server** (Port 8002)
   - Windows automation (file operations, Notepad, PDF reading)
   - URL: http://localhost:8002/mcp

5. **Screener API** (Port 8080)
   - Stock screening and financial data
   - URL: http://localhost:8080

## After Starting Backend

Once all backend servers are running, start the frontend:

```bash
cd frontend
npm run dev
```

Frontend will be available at: http://localhost:3000

## Troubleshooting

### If a port is already in use:
1. Run the stop script: `.\stop-all-servers.ps1`
2. Wait a few seconds
3. Run the start script again: `.\start-all-servers.ps1`

### If PowerShell script won't run:
You may need to enable script execution:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Manual server management:
If you need to start servers individually:
```bash
# Insurance RAG API
cd rag_server
python app.py

# Main RAG API
python main.py

# RAG MCP Server
python run_mcp.py

# Computer MCP Server
python run_computer_mcp.py

# Screener API
cd screener_server
python server.py
```

## Configuration

### RAG API Key
Set your RAG API key in `backend/.env`:
```
RAG_API_KEY=your_bearer_token_here
```

This key is used by both the Main RAG API (port 8000) and RAG MCP Server (port 8001) to authenticate with the Insurance RAG API (port 7860).

### Insurance RAG Server
Configure the Insurance RAG server in `backend/rag_server/.env`:
- Gemini API keys (1-25)
- Groq API keys (1-15)
- Pinecone configuration
- Model settings and parameters

See `backend/rag_server/.env.template` for all available options.

## Dependencies

Dependencies are automatically installed when you run `setup-venvs.ps1` or `setup-venvs.bat`.

If you need to manually install or update dependencies:

```bash
# Main RAG API + MCP Servers
.venv_main\Scripts\pip.exe install -r requirements.txt

# Insurance RAG API
rag_server\.venv_rag\Scripts\pip.exe install -r rag_server\requirements.txt

# Screener API
screener_server\.venv_screener\Scripts\pip.exe install -r screener_server\requirements.txt
```
