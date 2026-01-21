from typing import Optional
from fastmcp import FastMCP
import os
import subprocess
import tempfile
import time
import asyncio

# Computer/Notepad MCP Server on port 8002
COMPUTER_MCP_PORT = int(os.getenv("COMPUTER_MCP_PORT", "8002"))

mcp = FastMCP("computer-server")

# Try to import pyautogui for keyboard simulation
try:
    import pyautogui
    pyautogui.FAILSAFE = False  # Disable fail-safe for smoother operation
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    print("Warning: pyautogui not installed. Stream typing will not be available.")
    print("Install with: pip install pyautogui")

# Try to import win32 for window management
try:
    import win32gui
    import win32con
    import win32process
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False
    print("Warning: pywin32 not installed. Window management may not work reliably.")
    print("Install with: pip install pywin32")

# Try to import PyMuPDF for PDF reading
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("Warning: PyMuPDF not installed. PDF reading will not be available.")
    print("Install with: pip install pymupdf")


@mcp.tool(description="Read content from a text file. Provide the full file path.")
async def read_file(file_path: str) -> dict:
    """Read content from a text file."""
    try:
        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "content": None
            }
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            "success": True,
            "file_path": file_path,
            "content": content,
            "size_bytes": len(content.encode('utf-8'))
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "content": None
        }


@mcp.tool(description="Read and extract all text content from a PDF file. Returns raw text - ideal for bills, invoices, receipts, and small documents. Use file path like: C:\\Users\\nihaa\\Downloads\\bill.pdf")
async def read_pdf(file_path: str) -> dict:
    """
    Read a PDF file and extract all text content directly.
    
    Args:
        file_path: Full path to the PDF file (e.g., C:\\Users\\nihaa\\Downloads\\bill.pdf)
    
    Returns:
        Dictionary with success status, extracted text, and metadata
    """
    if not PYMUPDF_AVAILABLE:
        return {
            "success": False,
            "error": "PyMuPDF is not installed. Install with: pip install pymupdf",
            "content": None
        }
    
    try:
        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "content": None
            }
        
        if not file_path.lower().endswith('.pdf'):
            return {
                "success": False,
                "error": "File is not a PDF",
                "content": None
            }
        
        # Open and read PDF
        doc = fitz.open(file_path)
        
        text_content = []
        for page_num, page in enumerate(doc, 1):
            page_text = page.get_text()
            if page_text.strip():
                text_content.append(f"--- Page {page_num} ---\n{page_text}")
        
        full_text = "\n\n".join(text_content)
        
        metadata = {
            "page_count": len(doc),
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "file_size_bytes": os.path.getsize(file_path)
        }
        
        doc.close()
        
        return {
            "success": True,
            "file_path": file_path,
            "content": full_text,
            "metadata": metadata,
            "character_count": len(full_text)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "content": None
        }


@mcp.tool(description="Write content to a text file. Creates the file if it doesn't exist, overwrites if it does.")
async def write_file(file_path: str, content: str) -> dict:
    """Write content to a text file."""
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            "success": True,
            "file_path": file_path,
            "message": f"Successfully wrote {len(content)} characters to {file_path}",
            "size_bytes": len(content.encode('utf-8'))
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path
        }


@mcp.tool(description="Append content to a text file. Creates the file if it doesn't exist.")
async def append_file(file_path: str, content: str) -> dict:
    """Append content to a text file."""
    try:
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(content)
        
        return {
            "success": True,
            "file_path": file_path,
            "message": f"Successfully appended {len(content)} characters to {file_path}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path
        }


@mcp.tool(description="Open Notepad with optional content. If content is provided, it creates a temp file and opens it in Notepad.")
async def open_notepad(content: Optional[str] = None, file_path: Optional[str] = None) -> dict:
    """Open Notepad, optionally with content or a specific file."""
    try:
        if file_path:
            # Open existing file in Notepad
            subprocess.Popen(['notepad.exe', file_path])
            return {
                "success": True,
                "message": f"Opened {file_path} in Notepad"
            }
        elif content:
            # Create temp file with content and open in Notepad
            temp_file = tempfile.NamedTemporaryFile(
                mode='w', 
                suffix='.txt', 
                delete=False, 
                encoding='utf-8'
            )
            temp_file.write(content)
            temp_file.close()
            subprocess.Popen(['notepad.exe', temp_file.name])
            return {
                "success": True,
                "message": f"Opened Notepad with content in temp file: {temp_file.name}",
                "temp_file": temp_file.name
            }
        else:
            # Just open empty Notepad
            subprocess.Popen(['notepad.exe'])
            return {
                "success": True,
                "message": "Opened empty Notepad"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


async def bring_notepad_to_foreground(process_id: int, max_wait: float = 3.0) -> bool:
    """
    Wait for Notepad window to exist and bring it to foreground.
    
    Args:
        process_id: The process ID of the Notepad process
        max_wait: Maximum time to wait for window in seconds
    
    Returns:
        True if window was found and brought to foreground, False otherwise
    """
    if not WIN32_AVAILABLE:
        return False
    
    start_time = time.time()
    hwnd = None
    
    # Wait for the window to be created
    while time.time() - start_time < max_wait:
        def callback(h, windows):
            try:
                _, pid = win32process.GetWindowThreadProcessId(h)
                if pid == process_id and win32gui.IsWindowVisible(h):
                    windows.append(h)
            except:
                pass
            return True
        
        windows = []
        win32gui.EnumWindows(callback, windows)
        
        if windows:
            hwnd = windows[0]
            break
        
        await asyncio.sleep(0.1)
    
    if hwnd:
        # Bring window to foreground
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(hwnd)
            return True
        except:
            pass
    
    return False


@mcp.tool(description="Stream/type text into Notepad character by character, creating a live typing effect. Opens Notepad and types the content in real-time.")
async def stream_to_notepad(
    content: str, 
    typing_speed: Optional[float] = 0.005,
    word_by_word: Optional[bool] = False
) -> dict:
    """
    Stream text to Notepad with a typing effect.
    
    Args:
        content: The text to type into Notepad
        typing_speed: Delay between characters/words in seconds (default: 0.005)
        word_by_word: If True, types word by word instead of character by character
    """
    if not PYAUTOGUI_AVAILABLE:
        return {
            "success": False,
            "error": "pyautogui is not installed. Install with: pip install pyautogui"
        }
    
    try:
        # Step 1: Start Notepad and get the process
        proc = subprocess.Popen(['notepad.exe'])
        
        # Step 2 & 3: Wait for window and bring to foreground
        if WIN32_AVAILABLE:
            foreground_success = await bring_notepad_to_foreground(proc.pid)
            if foreground_success:
                await asyncio.sleep(0.2)  # Small delay after focusing
            else:
                # Fallback: just wait
                await asyncio.sleep(0.8)
        else:
            # No win32, just wait and hope
            await asyncio.sleep(0.8)
        
        # Step 4: Type the content
        if word_by_word:
            # Split by whitespace but preserve the whitespace
            words = content.split(' ')
            for i, word in enumerate(words):
                pyautogui.typewrite(word, interval=0.01) if word.isascii() else type_unicode(word)
                if i < len(words) - 1:
                    pyautogui.press('space')
                await asyncio.sleep(typing_speed)
        else:
            # Character by character typing
            for char in content:
                if char == '\n':
                    pyautogui.press('enter')
                elif char == '\t':
                    pyautogui.press('tab')
                elif char.isascii() and char.isprintable():
                    pyautogui.typewrite(char, interval=0)
                else:
                    # Handle unicode characters
                    type_unicode(char)
                await asyncio.sleep(typing_speed)
        
        return {
            "success": True,
            "message": f"Successfully streamed {len(content)} characters to Notepad",
            "characters_typed": len(content),
            "mode": "word_by_word" if word_by_word else "character_by_character"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def type_unicode(text: str):
    """Type unicode characters using clipboard method."""
    import pyperclip
    try:
        # Save current clipboard
        old_clipboard = pyperclip.paste()
    except:
        old_clipboard = ""
    
    # Copy text to clipboard and paste
    pyperclip.copy(text)
    pyautogui.hotkey('ctrl', 'v')
    
    # Restore clipboard (optional)
    try:
        pyperclip.copy(old_clipboard)
    except:
        pass


@mcp.tool(description="Type text into the currently focused window (whatever app is active). Useful for typing into any application.")
async def type_text(
    content: str, 
    typing_speed: Optional[float] = 0.005,
    press_enter: Optional[bool] = False
) -> dict:
    """
    Type text into the currently active/focused window.
    
    Args:
        content: The text to type
        typing_speed: Delay between characters in seconds (default: 0.005)
        press_enter: Whether to press Enter after typing
    """
    if not PYAUTOGUI_AVAILABLE:
        return {
            "success": False,
            "error": "pyautogui is not installed. Install with: pip install pyautogui"
        }
    
    try:
        for char in content:
            if char == '\n':
                pyautogui.press('enter')
            elif char == '\t':
                pyautogui.press('tab')
            elif char.isascii() and char.isprintable():
                pyautogui.typewrite(char, interval=0)
            else:
                type_unicode(char)
            await asyncio.sleep(typing_speed)
        
        if press_enter:
            pyautogui.press('enter')
        
        return {
            "success": True,
            "message": f"Successfully typed {len(content)} characters",
            "characters_typed": len(content)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool(description="List files in a directory. Optionally filter by extension.")
async def list_files(directory: str, extension: Optional[str] = None) -> dict:
    """List files in a directory."""
    try:
        if not os.path.exists(directory):
            return {
                "success": False,
                "error": f"Directory not found: {directory}",
                "files": []
            }
        
        files = []
        for item in os.listdir(directory):
            full_path = os.path.join(directory, item)
            if os.path.isfile(full_path):
                if extension is None or item.endswith(extension):
                    files.append({
                        "name": item,
                        "path": full_path,
                        "size_bytes": os.path.getsize(full_path)
                    })
        
        return {
            "success": True,
            "directory": directory,
            "files": files,
            "count": len(files)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "files": []
        }


def run_server(port: int = None):
    """Run the computer MCP server."""
    server_port = port or COMPUTER_MCP_PORT
    print(f"Starting Computer MCP Server on port {server_port}...")
    mcp.run(transport="streamable-http", port=server_port)


if __name__ == "__main__":
    run_server()
