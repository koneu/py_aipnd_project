import torch
from torch import nn
from torchvision import models

model = None
classes = None
cat_to_name = None

def load_model(checkpoint_path: str, DEVICE: torch.device = torch.device("cpu")):
    global model, classes, cat_to_name
    if model is not None:
        return  # already loaded, skip
    
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=DEVICE)
    classes = checkpoint["classes"]
    cat_to_name = checkpoint["cat_to_name"]

    hidden_units = checkpoint["best_params"]["hidden_units"]
    num_classes = len(classes)

    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(hidden_units, num_classes)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()