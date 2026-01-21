from pathlib import Path
from typing import Dict, List, Any
import json
from datetime import datetime


class SimpleMermaidGenerator:
    
    def __init__(self):
        pass
    
    def load_trees_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'trees' in data:
            return data['trees']
        elif 'nodes' in data:
            return [data]
        else:
            return []
    
    def sanitize_text(self, text: str, max_length: int = 25) -> str:
        if not text:
            return "Unknown"
        
        text = text.replace('"', "'").replace('\\n', ' ').replace('|', '-')
        
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
    
    def generate_mermaid_tree(self, tree: Dict[str, Any]) -> str:
        nodes = tree.get('nodes', {})
        
        if isinstance(nodes, list):
            nodes_dict = {}
            for node in nodes:
                nodes_dict[node['node_id']] = node
            nodes = nodes_dict
        
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
            question = self.sanitize_text(node_data.get('question', 'Unknown'))
            node_type = node_data.get('node_type', 'decision')
            
            if node_type == 'root':
                # Include reasoning and sources for root node
                reasoning = self.sanitize_text(node_data.get('reasoning', ''), 40)
                sources = node_data.get('source_references', [])
                source_text = f" | Sources: {', '.join(sources)}" if sources else ""
                shape = f'{clean_id}[["🏁 {question}<br/><i>{reasoning}</i>{source_text}"]]'
            elif node_type == 'leaf':
                # Leaf nodes as squares with answer, reasoning, and sources
                answer = self.sanitize_text(node_data.get('answer', 'Result'), 30)
                reasoning = self.sanitize_text(node_data.get('reasoning', ''), 25)
                sources = node_data.get('source_references', [])
                confidence = node_data.get('confidence', 0.0)
                
                source_text = ""
                if sources:
                    source_text = f"<br/>📄 {', '.join(sources)}"
                
                display_text = f"✅ {answer}<br/><i>{reasoning}</i>{source_text}<br/>Confidence: {confidence*100:.0f}%"
                shape = f'{clean_id}["{display_text}"]'
            else:
                # Decision nodes with reasoning and sources
                reasoning = self.sanitize_text(node_data.get('reasoning', ''), 35)
                sources = node_data.get('source_references', [])
                source_text = f" | 📄 {', '.join(sources)}" if sources else ""
                shape = f'{clean_id}{{"❓ {question}<br/><i>{reasoning}</i>{source_text}"}}'
            
            lines.append(f"    {shape}")
        
        # Third pass: generate connections using the mapping
        for node_id, node_data in nodes.items():
            clean_id = id_mapping[node_id]
            
            if node_data.get('left_child') and node_data['left_child'] in id_mapping:
                left_id = id_mapping[node_data['left_child']]
                lines.append(f"    {clean_id} -->|❌ No| {left_id}")
            
            if node_data.get('right_child') and node_data['right_child'] in id_mapping:
                right_id = id_mapping[node_data['right_child']]
                lines.append(f"    {clean_id} -->|✅ Yes| {right_id}")
        
        lines.extend([
            "",
            "%% Enhanced Styling",
            "classDef rootStyle fill:#ff6b6b,stroke:#333,stroke-width:4px,color:#000,font-weight:bold",
            "classDef decisionStyle fill:#4ecdc4,stroke:#333,stroke-width:3px,color:#000,font-weight:bold", 
            "classDef leafStyle fill:#ffa726,stroke:#333,stroke-width:3px,color:#000,font-weight:bold"
        ])
        
        # Fourth pass: apply styling using the mapping
        for node_id, node_data in nodes.items():
            clean_id = id_mapping[node_id]
            node_type = node_data.get('node_type', 'decision')
            
            if node_type == 'root':
                lines.append(f"class {clean_id} rootStyle")
            elif node_type == 'leaf':
                lines.append(f"class {clean_id} leafStyle")
            else:
                lines.append(f"class {clean_id} decisionStyle")
        
        return "\n".join(lines)
    
    def generate_all_trees(self, file_path: str) -> str:
        trees = self.load_trees_from_file(file_path)
        
        if not trees:
            return "flowchart TD\n    Empty[\"No trees found\"]"
        
        if len(trees) == 1:
            return self.generate_mermaid_tree(trees[0])
        
        lines = ["flowchart TD"]
        
        for i, tree in enumerate(trees):
            tree_name = tree.get('name', tree.get('question', f'Tree_{i+1}'))
            tree_name = self.sanitize_text(tree_name, 20)
            
            lines.append(f"    subgraph Tree{i+1} [\"{tree_name}\"]")
            
            tree_content = self.generate_mermaid_tree(tree)
            tree_lines = tree_content.split('\n')[1:]
            
            for line in tree_lines:
                if line.strip() and not line.startswith('%'):
                    modified_line = line
                    for node_id in tree.get('nodes', {}):
                        if isinstance(tree.get('nodes'), dict):
                            clean_old = self.sanitize_node_id(node_id)
                            clean_new = f"T{i+1}_{clean_old}"
                            modified_line = modified_line.replace(clean_old, clean_new)
                    lines.append(f"    {modified_line}")
            
            lines.append("    end")
            lines.append("")
        
        return "\n".join(lines)


def generate_raw_mermaid(file_path: str, output_file: str = None):
    generator = SimpleMermaidGenerator()
    
    print(f"🎨 Generating Mermaid diagram from: {Path(file_path).name}")
    
    mermaid_code = generator.generate_all_trees(file_path)
    
    if output_file is None:
        base_name = Path(file_path).stem
        output_file = f"mermaid_{base_name}.mmd"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(mermaid_code)
    
    print(f"✅ Saved raw Mermaid code to: {output_file}")
    
    print("\\n" + "="*60)
    print("📋 RAW MERMAID CODE (copy and paste this)")
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
        generate_raw_mermaid(file_path, output_file)
    else:
        print("Usage: python simple_mermaid.py <tree_file.json> [output_file.mmd]")
        print("Example: python simple_mermaid.py results/decision_trees/tree.json")
