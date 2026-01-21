# Stop All Backend Servers
# This script stops all backend servers by killing Python processes on specific ports

Write-Host "Stopping all backend servers..." -ForegroundColor Red
Write-Host "=" * 60

function Stop-ServerOnPort {
    param (
        [int]$Port,
        [string]$ServerName
    )
    
    Write-Host "Stopping $ServerName on port $Port..." -ForegroundColor Yellow
    
    try {
        $connections = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
        
        if ($connections) {
            foreach ($conn in $connections) {
                $process = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
                if ($process) {
                    Write-Host "  Killing process: $($process.ProcessName) (PID: $($process.Id))" -ForegroundColor Gray
                    Stop-Process -Id $process.Id -Force
                }
            }
            Write-Host "  $ServerName stopped successfully" -ForegroundColor Green
        } else {
            Write-Host "  No process found on port $Port" -ForegroundColor Gray
        }
    } catch {
        Write-Host "  Error stopping $ServerName : $_" -ForegroundColor Red
    }
}

# Stop all servers
Stop-ServerOnPort -Port 7860 -ServerName "Insurance RAG API"
Stop-ServerOnPort -Port 8000 -ServerName "Main RAG API"
Stop-ServerOnPort -Port 8001 -ServerName "RAG MCP Server"
Stop-ServerOnPort -Port 8002 -ServerName "Computer MCP Server"
Stop-ServerOnPort -Port 8080 -ServerName "Screener API"

Write-Host ""
Write-Host "=" * 60
Write-Host "All backend servers stopped!" -ForegroundColor Green
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
