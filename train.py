import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Optional, Callable

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

    correct = 0 # nuof correct predictions
    total = 0 # nuof total predictiosn


    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            # Keep track of total samples processed
            total += target.size(0)

    return correct / total