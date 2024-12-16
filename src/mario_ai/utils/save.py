import torch
import json

def export_nn_to_json(model, path):
    model_structure = {
        "architecture": model.__class__.__name__,
        "layers": []
    }
    for name, layer in model.named_children():
        model_structure["layers"].append({
            "layer_name": name,
            "layer_type": layer.__class__.__name__,
            "parameters": {k: v.tolist() if torch.is_tensor(v) else v for k, v in layer.state_dict().items()}
        })

    with open(path, "w") as json_file:
        json.dump(model_structure, json_file, indent=4)