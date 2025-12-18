import torch
from torchvision import models
import os
import onnx
import sys

def get_weights_attr(model_name: str):
    """
    Search torchvision.models for the correct weights attribute.
    e.g., searches for something that looks like 'MobileNet_V2_Weights'
    """
    search_str = f"{model_name.replace('_', '')}_weights"
    for attr in dir(models):
        if attr.lower().replace('_', '') == search_str:
            return attr
    return None

def export_model(model_name: str):
    print(f"--- Processing: {model_name} ---")
    
    try:
        # 1. Dynamically load the model function
        model_fn = getattr(models, model_name)
        
        # 2. Find and load the correct Weights
        attr_name = get_weights_attr(model_name)
        if not attr_name:
            # Fallback for models that might not follow the _Weights pattern
            print(f"⚠️  Could not find specific weights for {model_name}, trying default...")
            model = model_fn(pretrained=True)
        else:
            weights_enum = getattr(models, attr_name)
            model = model_fn(weights=weights_enum.DEFAULT)
            print(f"Successfully loaded {model_name} using {attr_name}.DEFAULT")
            
    except Exception as e:
        print(f"❌ Error loading model '{model_name}': {e}")
        return

    model.eval()
    dummy = torch.randn(1, 3, 224, 224)

    # 3. Setup paths (Targets the 'model' folder in project root)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    output_path = os.path.join(project_root, "model", f"{model_name}.onnx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Exporting to: {output_path}")

    # 4. Perform the Export
    with torch.no_grad():
        torch.onnx.export(
            model, 
            (dummy,), 
            output_path,
            export_params=True,
            opset_version=18,
            input_names=["input"],
            output_names=["output"], 
            dynamic_axes={
                "input": {0: "batch"}, 
                "output": {0: "batch"}
            }
        )
    
    # 5. Merge external data into a single file
    print("Merging data into a single .onnx file...")
    model_proto = onnx.load(output_path)
    onnx.save_model(
        model_proto, 
        output_path, 
        save_as_external_data=False 
    )

    # Cleanup .data file
    data_file = output_path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)
    
    size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ SUCCESS: {model_name}.onnx created ({size:.2f} MB)\n")

if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["mobilenet_v2"]
    
    for target in targets:
        export_model(target)