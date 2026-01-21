import os
import httpx
from PIL import Image
from json_to_flowchart import FlowchartMermaidGenerator


def draw_flowchart_with_kroki(mermaid_code: str, output_path: str):
    url = "https://kroki.io/"
    payload = {
        "diagram_source": mermaid_code,
        "diagram_type": "mermaid",
        "output_format": "png",
        "background_color": "white"
    }

    try:
        with httpx.Client() as client:
            response = client.post(url, json=payload, timeout=30)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                f.write(response.content)

            try:
                with Image.open(output_path) as img:
                    if img.mode in ("RGBA", "LA", "P"):
                        background = Image.new("RGB", img.size, (255, 255, 255))
                        if img.mode == "P":
                            img = img.convert("RGBA")
                        background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
                        img = background
                    elif img.mode != "RGB":
                        img = img.convert("RGB")

                    jpeg_path = output_path.replace('.png', '.jpg')
                    img.save(jpeg_path, 'JPEG', quality=95)

            except Exception as convert_error:
                print(f"      Warning: Could not convert to JPEG: {convert_error}")

    except Exception as e:
        raise Exception(f"Failed to generate PNG with Kroki: {e}")


def convert_json_to_flowchart(base_path: str):

    if not os.path.exists(base_path):
        print(f"Base path {base_path} does not exist")
        return

    print(f"📁 Processing timestamp folder: {os.path.basename(base_path)}")

    generator = FlowchartMermaidGenerator()
    json_files_found = 0

    if os.path.isdir(base_path):
        topics = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        print(f"Found {len(topics)} topic folders: {topics}")

        for topic in topics:
            topic_path = os.path.join(base_path, topic)
            print(f"\n  📂 Processing topic: {topic}")

            for file in os.listdir(topic_path):
                if file.endswith(".json"):
                    json_files_found += 1
                    json_file = os.path.join(topic_path, file)
                    mmd_file = json_file.replace(".json", ".mmd")
                    png_file = json_file.replace(".json", ".png")

                    print(f"    📄 Processing {file}")

                    try:
                        mermaid_code = generator.generate_flowchart(json_file)

                        with open(mmd_file, 'w', encoding='utf-8') as f:
                            f.write(mermaid_code)
                        print(f"      ✅ Saved Mermaid flowchart to {os.path.basename(mmd_file)}")

                        try:
                            draw_flowchart_with_kroki(mermaid_code, png_file)
                            jpeg_file = png_file.replace('.png', '.jpg')
                            print(f"      ✅ Successfully converted to {os.path.basename(png_file)} and {os.path.basename(jpeg_file)}")
                        except Exception as e:
                            print(f"      ❌ Failed to convert {os.path.basename(mmd_file)} to images")
                            print(f"         Error: {e}")

                    except Exception as e:
                        print(f"      ❌ Error processing {file}: {e}")

    if json_files_found == 0:
        print(f"  ⚠️  No JSON files found in {base_path}")
    else:
        print(f"  📊 Processed {json_files_found} JSON files")

    print(f"\n🎉 Flowchart conversion complete for {os.path.basename(base_path)}!")

if __name__ == "__main__":
    base_path = "../../results/decision_trees/trees/a749066c4fcddec8/2025-09-16_22-42-38"

    if os.path.exists(base_path):
        convert_json_to_flowchart(base_path)
    else:
        print(f"Base path {base_path} does not exist")