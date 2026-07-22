import re
import base64
import os

readme_path = "README.md"
assets_dir = "assets"

os.makedirs(assets_dir, exist_ok=True)

with open(readme_path, "r", encoding="utf-8") as f:
    content = f.read()

# Regex to find all base64 images
# Format: [imageX]: <data:image/png;base64,.....>
pattern = re.compile(r'\[(image\d+)\]:\s*<data:image/[^;]+;base64,([^>]+)>')

def replace_func(match):
    image_name = match.group(1)
    b64_data = match.group(2)
    
    # Save the image
    image_path = os.path.join(assets_dir, f"{image_name}.png")
    with open(image_path, "wb") as img_file:
        img_file.write(base64.b64decode(b64_data))
    
    # Return the new markdown link
    return f"[{image_name}]: {assets_dir}/{image_name}.png"

new_content = pattern.sub(replace_func, content)

with open(readme_path, "w", encoding="utf-8") as f:
    f.write(new_content)

print("README organized successfully.")
