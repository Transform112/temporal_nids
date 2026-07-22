from pathlib import Path
import re

kaggle_dir = Path(r"C:\Users\potato\Desktop\ids-v2 - Copy\kaggle")
target_path = "/kaggle/input/datasets/mysteriousavailable/ids-nf3-processed/ids-nf3-processed"

for py_file in kaggle_dir.glob("*.py"):
    content = py_file.read_text(encoding='utf-8')
    # Use regex to robustly replace INPUT = Path(...) regardless of spacing
    new_content = re.sub(r"INPUT\s*=\s*Path\(['\"].*?['\"]\)", f"INPUT = Path('{target_path}')", content)
    
    if new_content != content:
        py_file.write_text(new_content, encoding='utf-8')
        print(f"Updated {py_file.name}")
    else:
        print(f"No match found in {py_file.name}")
