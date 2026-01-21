import os
import httpx
from json_to_mermaid import SimpleMermaidGenerator


def draw_mermaid_with_kroki(mermaid_code: str, output_path: str):
    """
    Convert Mermaid syntax to PNG using Kroki API.
    
    Args:
        mermaid_code: The Mermaid diagram syntax
        output_path: Path where to save the PNG file
    """
    url = "https://kroki.io/"
    payload = {
        "diagram_source": mermaid_code,
        "diagram_type": "mermaid",
        "output_format": "png"
    }
    
    try:
        with httpx.Client() as client:
            response = client.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
                
    except Exception as e:
        raise Exception(f"Failed to generate PNG with Kroki: {e}")


def convert_json_to_mermaid(base_path: str):
    """
    Convert JSON decision trees to Mermaid diagrams and PNG images.
    
    Args:
        base_path: Path to the timestamp folder containing topic subfolders with JSON files
    """
    if not os.path.exists(base_path):
        print(f"Base path {base_path} does not exist")
        return
    
    print(f"📁 Processing timestamp folder: {os.path.basename(base_path)}")
    
    generator = SimpleMermaidGenerator()
    json_files_found = 0
    
    # Get all topic folders in the timestamp directory
    if os.path.isdir(base_path):
        topics = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        print(f"Found {len(topics)} topic folders: {topics}")
        
        for topic in topics:
            topic_path = os.path.join(base_path, topic)
            print(f"\n  � Processing topic: {topic}")
            
            # Look for JSON file in the topic folder
            for file in os.listdir(topic_path):
                if file.endswith(".json"):
                    json_files_found += 1
                    json_file = os.path.join(topic_path, file)
                    mmd_file = json_file.replace(".json", ".mmd")
                    png_file = json_file.replace(".json", ".png")
                    
                    print(f"    📄 Processing {file}")
                    
                    try:
                        mermaid_code = generator.generate_all_trees(json_file)
                        
                        with open(mmd_file, 'w', encoding='utf-8') as f:
                            f.write(mermaid_code)
                        print(f"      ✅ Saved Mermaid code to {os.path.basename(mmd_file)}")
                        
                        try:
                            draw_mermaid_with_kroki(mermaid_code, png_file)
                            print(f"      ✅ Successfully converted to {os.path.basename(png_file)}")
                        except Exception as e:
                            print(f"      ❌ Failed to convert {os.path.basename(mmd_file)} to image")
                            print(f"         Error: {e}")
                            
                    except Exception as e:
                        print(f"      ❌ Error processing {file}: {e}")
    
    if json_files_found == 0:
        print(f"  ⚠️  No JSON files found in {base_path}")
    else:
        print(f"  📊 Processed {json_files_found} JSON files")
    
    print(f"\n🎉 Conversion complete for {os.path.basename(base_path)}!")


# For backwards compatibility when run as script
if __name__ == "__main__":
    base_path = "../../results/decision_trees/trees/a749066c4fcddec8/2025-09-16_20-01-27"
    
    if os.path.exists(base_path):
        # folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        # print(f"Found {len(folders)} folders: {folders}")
        
        # for folder in folders:
        #     folder_path = os.path.join(base_path, folder)
        #     convert_json_to_mermaid(folder_path)
        convert_json_to_mermaid(base_path)
    else:
        print(f"Base path {base_path} does not exist")