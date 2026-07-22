import os
import shutil
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(r"C:\Users\potato\Desktop\ids-v2 - Copy")
KAGGLE_DIR = PROJECT_ROOT / "kaggle"
PROCESSED_DIR = PROJECT_ROOT / "laptop" / "processed"
GRAPHS_DIR = PROCESSED_DIR / "graphs"
WORKING_DIR = PROCESSED_DIR / "working"

def setup():
    # Ensure yamls and scaler are in the INPUT directory for the Kaggle scripts to find
    shutil.copy(PROJECT_ROOT / "feature_manifest.yaml", GRAPHS_DIR / "feature_manifest.yaml")
    shutil.copy(PROJECT_ROOT / "label_map.yaml", GRAPHS_DIR / "label_map.yaml")
    if (PROCESSED_DIR / "scaler.pkl").exists():
        shutil.copy(PROCESSED_DIR / "scaler.pkl", GRAPHS_DIR / "scaler.pkl")
    
    WORKING_DIR.mkdir(parents=True, exist_ok=True)

def patch_and_run(script_path):
    print(f"\n{'='*60}\nDRY RUNNING: {script_path.name}\n{'='*60}")
    code = script_path.read_text(encoding='utf-8')
    
    # 1. Patch paths (Windows paths with double backslashes)
    input_str = str(GRAPHS_DIR).replace("\\", "\\\\")
    working_str = str(WORKING_DIR).replace("\\", "\\\\")
    
    code = code.replace("INPUT   = Path('../dataset/graphs')", f"INPUT = Path('{input_str}')")
    code = code.replace("WORKING = Path('../working')", f"WORKING = Path('{working_str}')")
    code = code.replace("WORKING=Path('../working');INPUT=Path('../dataset/graphs')", f"WORKING=Path('{working_str}');INPUT=Path('{input_str}')")
    
    # 2. Patch epochs
    code = code.replace("'epochs':30", "'epochs':1")
    code = code.replace("'epochs': 30", "'epochs':1")
    code = code.replace("'epochs':100", "'epochs':1")
    code = code.replace("'epochs':50", "'epochs':1")
    
    # 3. Patch loops to break after 1 batch (for speed)
    code = code.replace("for bi, batch in enumerate(loader):", "for bi, batch in enumerate(loader):\n        if bi > 1: break")
    code = code.replace("for bi, batch in enumerate(train_loader):", "for bi, batch in enumerate(train_loader):\n        if bi > 1: break")
    code = code.replace("for step, batch in enumerate(loader):", "for step, batch in enumerate(loader):\n        if step > 1: break")
    code = code.replace("n_runs=200", "n_runs=2")
    
    # 4. Write temp file and execute
    temp_path = PROCESSED_DIR / f"temp_dry_{script_path.name}"
    temp_path.write_text(code, encoding='utf-8')
    
    result = subprocess.run(["python", str(temp_path)], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR OUTPUT:\n{result.stderr}")
        print(f"--- FAILED {script_path.name} ---")
        temp_path.unlink(missing_ok=True)
        return False
        
    temp_path.unlink(missing_ok=True)
    return True

if __name__ == "__main__":
    setup()
    scripts = sorted(KAGGLE_DIR.glob('k*.py'))
    success = True
    for k in scripts:
        if not patch_and_run(k):
            success = False
            break
            
    if success:
        print("\nALL KAGGLE SCRIPTS DRY-RUN PASSED!")
