import torch
import json
import os
from ai import neural_network

def export_nn_to_json(model, training_id, nb_gen, json_name, path = "saves/"):
    path += f"training{str(training_id)}/{str(nb_gen)}/" + json_name
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

def load_nn_from_json(path):
    with open(path, "r") as json_file:
        model_structure = json.load(json_file)
    
    if model_structure["architecture"] != "NN":
        raise ValueError(f"Architecture non supportée : {model_structure['architecture']}")
    
    model = neural_network.NN()
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

def new_training_folder(training_id, path = "saves/"):
    if not os.path.exists(path):
        os.makedirs(path)
    
    training_folder_path = os.path.join(path, "training"+str(training_id))
    if not os.path.exists(training_folder_path):
        os.makedirs(training_folder_path)
    return 0

def new_generation_folder(training_id, nb_gen, path = "saves/"):
    if not os.path.exists(path):
        os.makedirs(path)
    
    generation_folder_path = os.path.join(path, "training"+str(training_id)+"/"+str(nb_gen))
    if not os.path.exists(generation_folder_path):
        os.makedirs(generation_folder_path)
    return 0
