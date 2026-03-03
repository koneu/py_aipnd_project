import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Optional, Callable
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision import models
import torch.optim as optim
import sys
import os
from typing import Optional, Callable
import json
import argparse

def train_model(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        criterion: Optional[Callable] = None
) -> float:
    """
    Trains the model for a single epoch.
    """
    
    model.train()
    total_loss = 0.0

    criterion = criterion or torch.nn.CrossEntropyLoss()
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)

        # vorwärts
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # rückwärts
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def eval_model(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device
) -> float:
    """
    Evaluates the model for a single epoch.  # fix copy paste error
    """
    
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0

    criterion = torch.nn.CrossEntropyLoss()


    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Calculate Loss
            loss = criterion(output, target)
            total_loss += loss.item()
            
            # Calculate Accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return accuracy, avg_loss

if __name__ == "__main__":
    CHECKPOINT_DIR = "checkpoints"
    BATCHSIZE = 64

    # default values from opt_study; expecting 90% accuracy
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cpu or cuda)')
    parser.add_argument('--lr', type=float, default=0.0005008419921592262, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--hu', type=int, default=2454, help='Number of hidden units')
    parser.add_argument('--do', type=float, default=0.38697266011799636, help='Dropout rate')
    args = parser.parse_args()
    optimizer_name = "Adam"

    # Check availability and user preference
    hidden_units = args.hu
    if args.device == "cpu" or not torch.cuda.is_available():
        DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device("cuda")

    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Data Transformer
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    dataset_train = ImageFolder(root=train_dir, transform=data_transforms)
    dataset_valid = ImageFolder(root=valid_dir, transform=data_transforms)
    dataset_test = ImageFolder(root=test_dir, transform=data_transforms)
    # only got one label per image so no one-hot encoding

    # Load the datasets with ImageFolder
    train_loader = DataLoader(dataset_train, batch_size=BATCHSIZE, shuffle=True)
    val_loader = DataLoader(dataset_valid, batch_size=BATCHSIZE, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=BATCHSIZE, shuffle=False)

    # --- Load pretrained ResNet50 -----------------------------------
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # freeze all pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier layer with our own
    num_classes = len(dataset_train.classes)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(p=args.do),
        nn.Linear(hidden_units, num_classes)
    )
    model = model.to(DEVICE)

    lr = args.lr
    optimizer = getattr(optim, optimizer_name)(model.fc.parameters(), lr=lr)

    for epoch in range(args.epochs):
        loss = train_model(model, train_loader, optimizer, DEVICE)
        eaccuracy, eloss = eval_model(model, val_loader, DEVICE)
        print("---------------------")
        print(f"Epoch:         {epoch}")
        print(f"loss:          {loss}")
        print(f"Eval loss:     {eloss}")
        print(f"Eval accuracy: {eaccuracy}")


    accuracy_loss = eval_model(model, test_loader, DEVICE)
    print("=====================")
    print(f"Test set accuracy, loss: {accuracy_loss}")
    print("=====================")

    with open("cat_to_name.json", "r") as f:
        cat_to_name = json.load(f)
        torch.save({
            "model_state_dict": model.state_dict(),
            "best_params": None,
            "best_value": None,
            "classes": dataset_train.classes,
            "epoch": epoch,
            "cat_to_name": cat_to_name,      
        }, os.path.join(CHECKPOINT_DIR, "model_trial_dummy.pth"))

        
