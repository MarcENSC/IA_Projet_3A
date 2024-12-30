import torch
import json
import os
from ai import neural_network

def export_nn_to_json(ind, training_type, training_id, nb_gen, json_name, path = "saves/"):
    if not os.path.exists(path):
        os.makedirs(path)
    
    path += f"{training_type}/"
    if not os.path.exists(path):
        os.makedirs(path)
    
    path += f"training{str(training_id)}/"
    if not os.path.exists(path):
        os.makedirs(path)
    
    path += f"{str(nb_gen)}/"
    if not os.path.exists(path):
        os.makedirs(path)

    path += json_name

    model = ind.neural_network

    model_structure = {
        "architecture": model.__class__.__name__,
        "id": ind.id,
        "score": ind.score,
        "nn_format": model.nn_format,
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

def load_nn_from_json(path):
    with open(path, "r") as json_file:
        model_structure = json.load(json_file)
    
    if model_structure["architecture"] != "NN":
        raise ValueError(f"Architecture non supportée : {model_structure['architecture']}")
    
    model = neural_network.NN(model_structure["nn_format"])
    layers = dict(model.named_children())
    
    for layer_info in model_structure["layers"]:
        layer_name = layer_info["layer_name"]
        if layer_name in layers:
            layer = layers[layer_name]
            state_dict = {k: torch.tensor(v) for k, v in layer_info["parameters"].items()}
            layer.load_state_dict(state_dict)
        else:
            raise ValueError(f"Couche {layer_name} non trouvée dans le modèle.")
    
    return model