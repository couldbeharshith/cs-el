# Start the backend

Install `uv`, configure the repository-root `.env`, then run this from the repository root:

```powershell
uv run --script backend/serve.py
```

The launcher prepares dependencies and starts the gateway, document MCP, insurance worker, screener, and Windows computer MCP. Wait for `Backend ready.` and press Ctrl+C in the same terminal to stop the stack.

See the [backend README](README.md) for credentials, port settings, and options for a remote worker or a smaller local stack.
