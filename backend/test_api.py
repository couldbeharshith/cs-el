import asyncio
import aiohttp
import json
import os
import importlib.util
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, IntPrompt
from rich.progress import Progress, SpinnerColumn, TextColumn

def load_test_data_from_file(file_path):
    if file_path.suffix == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        spec = importlib.util.spec_from_file_location("test_module", file_path)
        test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_module)
        return test_module.TEST_DATA

def discover_test_files():
    tests_dir = Path("tests")
    test_files = []
    
    if tests_dir.exists():
        for file_path in list(tests_dir.glob("*.py")) + list(tests_dir.glob("*.json")):
            if file_path.name == "test_config.py":
                continue
            try:
                test_data = load_test_data_from_file(file_path)
                test_files.append({
                    "file_path": str(file_path),
                    "name": test_data.get("name", file_path.stem),
                    "description": test_data.get("description", f"Test file: {file_path.stem}"),
                    "data": test_data
                })
            except Exception as e:
                print(f"⚠️ Error loading {file_path}: {e}")

    return test_files

async def save_results_to_markdown(result, test_data, test_file_name, console, static_k: int | None = None):
    """Saves the test results to a markdown file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    raw_response = result.get('raw_response', {})
    retrieval_method = raw_response.get('retrieval_method', 'N/A')
    processing_mode = raw_response.get('processing_mode', 'traditional')

    results_dir = f"results/{test_file_name}"
    os.makedirs(results_dir, exist_ok=True)
    filename = f"{results_dir}/{processing_mode}_{retrieval_method.replace(' ', '_').replace('+', 'plus')}_{timestamp}.md"

    md_content = f"""# RAG Test Results

## Test Information
- **Test File**: `{test_file_name}`
- **Timestamp**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Pipeline**: `{processing_mode.replace('_', ' ').title()} RAG`
- **Retrieval Method**: `{retrieval_method}`
- **Processing Time**: {result.get('processing_time', 'N/A'):.2f} seconds
- **Total Questions**: {len(test_data.get('questions', []))}

---

## Questions & Answers
"""
    answers = result.get('answers', [])
    debug_info = raw_response.get('debug_info', [])

    for i, (question, answer) in enumerate(zip(test_data.get('questions', []), answers), 1):
        md_content += f"\n### Q{i}: {question}\n\n**Answer**: {answer}\n"
        
        if i-1 < len(debug_info):
            debug = debug_info[i-1]
            # Prefer dynamic_k/static_k from backend; fall back to user-provided static_k
            if 'dynamic_k' in debug:
                k_val = debug['dynamic_k']
            elif 'static_k' in debug:
                k_val = debug['static_k']
            elif static_k is not None:
                k_val = static_k
            else:
                k_val = 'N/A'
            md_content += f"\n> Chunks Used: {k_val}\n"

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(md_content)
    console.print(f"📄 Results saved to: [cyan]{filename}[/cyan]")

async def run_single_test(session, base_url, headers, test_info, console, api_data_override):
    """Runs a single test case against the API."""
    test_data = test_info["data"]
    test_name = test_info["name"]

    console.print(Panel(f"[bold]🧪 {test_name}[/bold]\n[dim]Questions: {len(test_data.get('questions', []))}[/dim]", border_style="blue"))
    
    try:
        api_data = test_data.copy()
        api_data.setdefault('delete_after', True)
        
        # --- HANDLE K PARAMETER FOR STATIC VS DYNAMIC TESTING ---
        # This logic works for any pipeline: remove k for dynamic, set k for static
        if 'k' not in api_data_override:
            api_data.pop('k', None)  # Dynamic mode - remove k to trigger dynamic behavior
        else:
            # Static mode - apply the override k value
            api_data.update(api_data_override)
        
        # Apply pipeline mode from override FIRST, then display
        if 'processing_mode' in api_data_override:
            api_data['processing_mode'] = api_data_override['processing_mode']

        retrieval_mode = "Dynamic K (Adaptive)" if 'k' not in api_data else f"Static K (k={api_data.get('k')})"
        pipeline_mode = api_data.get('processing_mode', 'traditional').replace('_', ' ').title()
        console.print(f"[bold yellow]K Mode:[/bold yellow] {retrieval_mode}")
        console.print(f"[bold yellow]Pipeline:[/bold yellow] {pipeline_mode} RAG")

        async with session.post(f"{base_url}/rag/run", headers=headers, json=api_data, timeout=aiohttp.ClientTimeout(total=1200)) as response:
            if response.status == 200:
                result = await response.json()
                console.print(Panel(f"[bold green]✅ Success![/bold green]\n[dim]⏱️  {result.get('processing_time', 0):.2f}s[/dim]", border_style="green"))
                
                answers = result.get('answers', [])
                raw_response = result.get('raw_response', {})
                debug_info = raw_response.get('debug_info', [])

                for i, (question, answer) in enumerate(zip(test_data.get('questions', []), answers), 1):
                    console.print(Panel(f"[bold cyan]Q{i}:[/bold cyan] {question}", title=f"Question {i}", border_style="blue"))
                    
                    chunk_info = ""
                    if i-1 < len(debug_info):
                        debug = debug_info[i-1]
                        if 'dynamic_k' in debug:
                            chunk_info = f"\n\n[dim]Chunks Used (Dynamic): {debug['dynamic_k']}[/dim]"
                        elif 'static_k' in debug:
                             chunk_info = f"\n\n[dim]Chunks Used (Static): {debug['static_k']}[/dim]"
                        elif 'k' in api_data:
                             chunk_info = f"\n\n[dim]Chunks Used (Static): {api_data['k']}[/dim]"
                    
                    console.print(Panel(f"{answer}{chunk_info}", title=f"Answer {i}", title_align="left", border_style="green"))

                file_name = Path(test_info["file_path"]).stem
                await save_results_to_markdown(result, test_data, file_name, console, static_k=api_data.get('k'))
                return True
            else:
                error_text = await response.text()
                console.print(Panel(f"[bold red]❌ Failed: {response.status}[/bold red]\n[red]{error_text}[/red]", border_style="red"))
                return False
    except Exception as e:
        console.print(Panel(f"[bold red]❌ Exception: {str(e)}[/bold red]", border_style="red"))
        return False

async def test_rag_api_modular(selected_tests, api_data_override):
    """Orchestrates the execution of a list of selected tests."""
    console = Console()
    base_url = "http://localhost:8000"
    headers = {
        "Authorization": "Bearer 78fb5a798235f5f9f659aa6e26fbece2b49413756a8fc4ee51ee190e89232496",
        "Content-Type": "application/json"
    }
    
    async with aiohttp.ClientSession() as session:
        successful_tests, failed_tests = 0, 0
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("[cyan]Running tests...", total=len(selected_tests))
            for test_info in selected_tests:
                success = await run_single_test(session, base_url, headers, test_info, console, api_data_override)
                if success: successful_tests += 1
                else: failed_tests += 1
                progress.update(task, advance=1)
                if len(selected_tests) > 1:
                    await asyncio.sleep(1)
        
        results_table = Table(title="[bold]📊 Test Results Summary[/bold]")
        results_table.add_column("Status", style="bold", justify="center")
        results_table.add_column("Count", style="bold cyan", justify="center")
        results_table.add_row("✅ Passed", str(successful_tests))
        results_table.add_row("❌ Failed", str(failed_tests))
        console.print("\n", results_table)

if __name__ == "__main__":
    console = Console()
    console.print(Panel("[bold magenta]🔬 RAG API Test Script[/bold magenta]\n[dim]CLI for testing your RAG endpoints[/dim]", border_style="magenta"))
    test_files = discover_test_files()

    if not test_files:
        console.print(Panel("[bold red]❌ No test files found in 'tests/' folder.[/bold red]", border_style="red"))
        exit(1)

    tests_table = Table(title="[bold]📁 Available Tests[/bold]", show_header=True, header_style="bold blue")
    tests_table.add_column("#", style="cyan", justify="center")
    tests_table.add_column("Test Name", style="white")
    tests_table.add_column("Questions", style="green", justify="center")
    for i, test in enumerate(test_files, 1):
        tests_table.add_row(str(i), test['name'], str(len(test["data"].get('questions', []))))
    console.print(tests_table)

    console.print("\n[bold]Select tests to run:[/bold] ([dim]e.g., 1,3,5 or all or q to quit[/dim])")
    selection = Prompt.ask("[cyan]Your choice[/cyan]").strip().lower()

    if selection == 'q': exit()
    elif selection == 'all':
        selected_tests = test_files
    else:
        try:
            indices = [int(x.strip()) - 1 for x in selection.split(',')]
            selected_tests = [test_files[i] for i in indices if 0 <= i < len(test_files)]
        except (ValueError, IndexError):
            console.print(Panel("[bold red]❌ Invalid selection![/bold red]", border_style="red")); exit(1)
    
    if not selected_tests:
        console.print(Panel("[bold red]❌ No valid tests selected![/bold red]", border_style="red")); exit(1)

    console.print("\n[bold]Select Retrieval Mode:[/bold]")
    console.print("  [bold]1[/bold]. [green]Dynamic K[/green] (Adaptive Chunk Selection) [dim]- Recommended for accuracy[/dim]")
    console.print("  [bold]2[/bold]. [green]Static K[/green] (Fixed Chunk Count) [dim]- Consistent chunk usage[/dim]")
    mode_choice = Prompt.ask("[cyan]Mode[/cyan]", choices=["1", "2"], default="1")

    api_data_override = {}
    if mode_choice == "2":
        static_k = IntPrompt.ask("[cyan]Enter the number of chunks (k) to retrieve[/cyan]", default=10)
        api_data_override['k'] = static_k

    console.print("\n[bold]Select RAG Pipeline:[/bold]")
    
    # Dynamically get available pipelines
    try:
        from app.services.pipelines.pipeline_manager import PipelineManager
        available_pipelines = PipelineManager.get_supported_pipelines()
    except ImportError:
        # Fallback if import fails
        available_pipelines = {
            "traditional": "Traditional RAG - Hybrid search + reranking with standard processing",
            "structure_aware": "Structure-Aware RAG - Enhanced processing for insurance documents with table/cell-aware reranking"
        }
    
    # Display pipeline options
    pipeline_choices = []
    for i, (pipeline_key, description) in enumerate(available_pipelines.items(), 1):
        display_name = pipeline_key.replace('_', ' ').title()
        console.print(f"  [bold]{i}[/bold]. [blue]{display_name} RAG[/blue] [dim]- {description}[/dim]")
        pipeline_choices.append(str(i))
    
    pipeline_choice = Prompt.ask("[cyan]Pipeline[/cyan]", choices=pipeline_choices, default="1")
    
    # Map choice to pipeline key
    pipeline_keys = list(available_pipelines.keys())
    selected_pipeline = pipeline_keys[int(pipeline_choice) - 1]
    
    if selected_pipeline != "traditional":  # Only set if not default
        api_data_override['processing_mode'] = selected_pipeline

    # Display final execution message
    selected_pipeline_name = api_data_override.get('processing_mode', 'traditional').replace('_', ' ').title()
    console.print(Panel(f"[bold blue]🚀 Starting {len(selected_tests)} selected test(s) with {selected_pipeline_name} RAG...[/bold blue]", border_style="blue"))
    asyncio.run(test_rag_api_modular(selected_tests, api_data_override))