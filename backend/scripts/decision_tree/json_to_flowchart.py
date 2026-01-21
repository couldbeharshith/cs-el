from pathlib import Path
from typing import Dict, List, Any
import json
from datetime import datetime


class FlowchartMermaidGenerator:

    def __init__(self):
        pass

    def load_flowchart_from_file(self, file_path: str) -> Dict[str, Any]:
        """Load flowchart data from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def sanitize_text(self, text: str, max_length: int = 1000) -> str:
        """Sanitize text for Mermaid compatibility."""
        if not text:
            return "Unknown"

        # Replace problematic characters for Mermaid
        text = text.replace('"', "'").replace('\\n', ' ').replace('|', '-')
        text = text.replace('<', '&lt;').replace('>', '&gt;')  # Escape HTML tags
        text = text.replace('(', '（').replace(')', '）')  # Replace parentheses with full-width versions
        text = text.replace('{', '｛').replace('}', '｝')  # Replace curly braces with full-width versions
        
        # Don't truncate by default, only if explicitly requested
        if max_length and len(text) > max_length:
            text = text[:max_length-3] + "..."

        return text

    def sanitize_node_id(self, node_id: str, used_ids: set = None) -> str:
        """Sanitize node ID to be Mermaid-compatible and ensure uniqueness."""
        if used_ids is None:
            used_ids = set()

        # Replace problematic characters
        clean_id = node_id.replace('-', '_').replace(' ', '_').replace('.', '_')

        # If ID is too long, try to make it shorter but unique
        if len(clean_id) > 15:
            # Take first 8 chars + last 6 chars to preserve uniqueness
            clean_id = clean_id[:8] + '_' + clean_id[-6:]

        # Ensure uniqueness by adding suffix if needed
        original_id = clean_id
        counter = 1
        while clean_id in used_ids:
            clean_id = f"{original_id}_{counter}"
            counter += 1

        used_ids.add(clean_id)
        return clean_id

    def generate_mermaid_flowchart(self, flowchart_data: Dict[str, Any]) -> str:
        """Generate Mermaid flowchart syntax from flowchart data."""
        nodes = flowchart_data.get('nodes', {})

        if not nodes:
            return "flowchart TD\n    Empty[\"No nodes found\"]"

        lines = ["flowchart TD"]
        

        used_ids = set()  # Track used IDs to ensure uniqueness
        id_mapping = {}  # Map original node IDs to sanitized IDs

        # First pass: create mapping of all node IDs
        for node_id in nodes.keys():
            clean_id = self.sanitize_node_id(node_id, used_ids)
            id_mapping[node_id] = clean_id

        # Second pass: generate node definitions with enhanced content
        for node_id, node_data in nodes.items():
            clean_id = id_mapping[node_id]
            node_type = node_data.get('node_type', 'process')
            title = self.sanitize_text(node_data.get('title', 'Unknown'), 200)
            description = self.sanitize_text(node_data.get('description', ''), 300)

            if node_type == 'start':
                # Start nodes as rounded rectangles with green color
                shape = f'{clean_id}(["🚀 {title}<br/><i>{description}</i>"])'
            elif node_type == 'condition':
                # Condition nodes as diamonds
                shape = f'{clean_id}{{"❓ {title}<br/><i>{description}</i>"}}'
            elif node_type == 'outcome':
                # Outcome nodes as rectangles with result info
                answer = self.sanitize_text(node_data.get('answer', 'Result'), 200)
                reasoning = self.sanitize_text(node_data.get('reasoning', ''), 300)
                confidence = node_data.get('confidence', 0.0)

                display_text = f"✅ {answer}<br/><i>{reasoning}</i><br/>Confidence: {confidence*100:.0f}%"
                shape = f'{clean_id}["{display_text}"]'
            elif node_type == 'process':
                # Process nodes as rectangles with process info
                shape = f'{clean_id}["⚙️ {title}<br/><i>{description}</i>"]'
            else:
                # Default shape
                shape = f'{clean_id}["{title}<br/><i>{description}</i>"]'

            lines.append(f"    {shape}")

        # Third pass: generate connections
        for node_id, node_data in nodes.items():
            clean_id = id_mapping[node_id]
            node_type = node_data.get('node_type', 'process')

            if node_type == 'condition':
                # Handle branches for condition nodes
                branches = node_data.get('branches', [])
                for branch in branches:
                    next_node = branch.get('next_node')
                    if next_node and next_node in id_mapping:
                        condition = branch.get('condition', 'yes')
                        label = self.sanitize_text(branch.get('label', condition), 100)
                        next_clean_id = id_mapping[next_node]
                        lines.append(f"    {clean_id} -->|{label}| {next_clean_id}")
            else:
                # Handle next_nodes for other node types
                next_nodes = node_data.get('next_nodes', [])
                for next_node in next_nodes:
                    if next_node in id_mapping:
                        next_clean_id = id_mapping[next_node]
                        lines.append(f"    {clean_id} --> {next_clean_id}")

        # Add styling
        lines.extend([
            "",
            "%% Enhanced Styling",
            "classDef startStyle fill:#4CAF50,stroke:#2E7D32,stroke-width:3px,color:#fff,font-weight:bold",
            "classDef conditionStyle fill:#FF9800,stroke:#E65100,stroke-width:3px,color:#000,font-weight:bold",
            "classDef outcomeStyle fill:#2196F3,stroke:#0D47A1,stroke-width:3px,color:#fff,font-weight:bold",
            "classDef processStyle fill:#9C27B0,stroke:#4A148C,stroke-width:3px,color:#fff,font-weight:bold"
        ])

        # Fourth pass: apply styling using the mapping
        for node_id, node_data in nodes.items():
            clean_id = id_mapping[node_id]
            node_type = node_data.get('node_type', 'process')

            if node_type == 'start':
                lines.append(f"class {clean_id} startStyle")
            elif node_type == 'condition':
                lines.append(f"class {clean_id} conditionStyle")
            elif node_type == 'outcome':
                lines.append(f"class {clean_id} outcomeStyle")
            elif node_type == 'process':
                lines.append(f"class {clean_id} processStyle")

        return "\n".join(lines)

    def generate_flowchart(self, file_path: str) -> str:
        """Generate Mermaid flowchart from JSON file."""
        flowchart_data = self.load_flowchart_from_file(file_path)
        return self.generate_mermaid_flowchart(flowchart_data)


def generate_raw_flowchart(file_path: str, output_file: str = None):
    """Generate raw Mermaid flowchart from JSON file."""
    generator = FlowchartMermaidGenerator()

    print(f"🎨 Generating Mermaid flowchart from: {Path(file_path).name}")

    mermaid_code = generator.generate_flowchart(file_path)

    if output_file is None:
        base_name = Path(file_path).stem
        output_file = f"flowchart_{base_name}.mmd"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(mermaid_code)

    print(f"✅ Saved raw Mermaid flowchart to: {output_file}")

    print("\\n" + "="*60)
    print("📋 MERMAID FLOWCHART CODE")
    print("="*60)
    print(mermaid_code)
    print("="*60)
    print("\\n💡 You can paste this code into:")
    print("   • GitHub markdown files")
    print("   • Mermaid Live Editor (https://mermaid.live/)")
    print("   • VS Code with Mermaid extension")
    print("   • Notion, GitLab, etc.")

    return mermaid_code


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        generate_raw_flowchart(file_path, output_file)
    else:
        print("Usage: python json_to_flowchart.py <flowchart_file.json> [output_file.mmd]")
        print("Example: python json_to_flowchart.py flowchart_data.json")