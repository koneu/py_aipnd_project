import torch
from torch import nn
import numpy as np
from typing import Tuple, List, Union
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
import sys

def process_image(image: Union[str, Image.Image]) -> np.ndarray:
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array.
    '''                
    # If a string path is passed, open the image
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # missing normalization!
    ])

    return transform(image).unsqueeze(0)
    

def predict(
    image_path: Union[str, Path], 
    model: nn.Module, 
    topk: int = 5
) -> Tuple[List[float], List[str]]:
    model.eval()
    image = process_image(image_path)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        probs, indices = torch.topk(probs, topk)

    return probs.squeeze().tolist(), indices.squeeze().tolist()


#static,s o we don't reload model etc. every lookup
_model = None
_classes = None
_cat_to_name = None

def _load_model(checkpoint_path: str):
    global _model, _classes, _cat_to_name
    if _model is not None:
        return  # already loaded, skip
    
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    _classes = checkpoint["classes"]
    _cat_to_name = checkpoint["cat_to_name"]
    
    _model = models.resnet50(weights=None)
    _model.fc = nn.Linear(_model.fc.in_features, len(_classes))
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.eval()


def predict_class(checkpoint_path: str, image_path: str, topk: int = 5) -> list[tuple[str, float]]:
    _load_model(checkpoint_path)
    probs, indices = predict(image_path, _model, topk)
    classes = [_classes[i] for i in indices]
    return list(zip(classes, probs))

def predict_label(checkpoint_path: str, image_path: str, topk: int = 5) -> list[tuple[str, float]]:
    results = predict_class(checkpoint_path, image_path, topk)
    return [(_cat_to_name[cls], prob) for cls, prob in results]


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

if __name__ == "__main__":
    checkpoint_path = "checkpoints/best_model.pth"
    fallback_image = "flowers/test/1/image_06743.jpg"
    topk = 5

    image_path = sys.argv[1] if len(sys.argv) > 1 else fallback_image

    results = predict_label(checkpoint_path, image_path, topk=topk)
    print(results)

    # Get tensor image for imshow
    tensor_image = process_image(image_path).squeeze(0)
    labels = [r[0] for r in results]
    probs = [r[1] for r in results]

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10))

    imshow(tensor_image, ax=ax1)
    ax1.set_title(f"{labels[0]}", fontsize=14)
    ax1.axis("off")

    ax2.barh(labels[::-1], probs[::-1])
    ax2.set_xlabel("Probability")
    ax2.set_xlim(0, 1)
    ax2.set_title("Top Predictions")

    plt.tight_layout()
    plt.savefig("prediction.png")
    plt.show()

    

