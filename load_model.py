import torch
from torch import nn
from torchvision import models

model = None
classes = None
cat_to_name = None

def load_model(checkpoint_path: str):
    global model, classes, cat_to_name
    if model is not None:
        return  # already loaded, skip
    
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    classes = checkpoint["classes"]
    cat_to_name = checkpoint["cat_to_name"]
    
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()