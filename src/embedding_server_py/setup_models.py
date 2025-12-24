import os
import shutil
from pathlib import Path
import sys

# Add the current directory to path to ensure we can import if needed, 
# though we are just using libraries here.

try:
    from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTModelForZeroShotImageClassification
    from transformers import AutoProcessor, AutoTokenizer
except ImportError:
    print("Error: Required libraries not installed. Please run 'pip install -r requirements.txt'")
    sys.exit(1)

# Configuration
# Using SigLIP 1 as a safe default for "siglip2" request if siglip2 is not yet available/supported by optimum.
# You can change the ID to "google/siglip2-base-patch16-512" if it becomes available.
MODELS = {
    "siglip2": {
        # "id": "google/siglip-base-patch16-224", 
        "id": "google/siglip2-base-patch16-512", 
        "task": "zero-shot-image-classification",
        "folder": "models/siglip2"
    },
    "dinov2": {
        "id": "facebook/dinov2-base",
        "task": "feature-extraction",
        "folder": "models/dinov2"
    }
}

def export_model(model_key, config):
    print(f"Processing {model_key}...")
    output_dir = Path(config["folder"])
    
    # Check if model already exists (simple check)
    if output_dir.exists() and (output_dir / "model.onnx").exists():
        print(f"Model {model_key} seems to be already exported at {output_dir}. Skipping.")
        return
    
    # For SigLIP/CLIP, it might export multiple files
    if output_dir.exists() and (output_dir / "vision_model.onnx").exists():
        print(f"Model {model_key} seems to be already exported at {output_dir}. Skipping.")
        return

    print(f"Exporting {config['id']} to ONNX...")
    
    try:
        if config["task"] == "feature-extraction":
            # For DinoV2
            model = ORTModelForFeatureExtraction.from_pretrained(config["id"], export=True)
            processor = AutoProcessor.from_pretrained(config["id"])
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)
            
        elif config["task"] == "zero-shot-image-classification":
            # For SigLIP
            model = ORTModelForZeroShotImageClassification.from_pretrained(config["id"], export=True)
            processor = AutoProcessor.from_pretrained(config["id"])
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)
            
        print(f"Successfully exported {model_key} to {output_dir}")
        
    except Exception as e:
        print(f"Failed to export {model_key}: {e}")
        # Clean up partial download/export
        if output_dir.exists():
            shutil.rmtree(output_dir)

if __name__ == "__main__":
    # Ensure models directory exists in the root or relative to this script
    # Assuming this script is run from project root or src/embedding_server_py
    # We want 'models' folder to be in the same directory as main_es.py usually, 
    # or in the project root. 
    # main_es.py has MODEL_PATHS = {"siglip2": "models/siglip2.onnx", ...}
    # We will change main_es.py to point to the folders.
    
    # Let's determine the base path. 
    # If run from root (e:\ImageCleverSearcherEmbedding), we want src/embedding_server_py/models
    # Or just models/ in root?
    # main_es.py is in src/embedding_server_py/
    # It refers to "models/siglip2.onnx".
    # So it expects a 'models' folder in the CWD when running main_es.py.
    # If we run main_es.py from root, it expects 'models' in root.
    # If we run main_es.py from src/embedding_server_py/, it expects 'models' there.
    
    # The install.ps1 says: "python src/embedding_server_py/main_es.py"
    # So CWD is likely the project root.
    # So 'models' should be in project root.
    
    base_dir = Path("models")
    base_dir.mkdir(exist_ok=True)
    
    for key, config in MODELS.items():
        # Update folder path to be relative to CWD
        config["folder"] = str(base_dir / key)
        export_model(key, config)
