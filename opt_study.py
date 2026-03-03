import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import optuna
from train import train_model, eval_model
import os
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {DEVICE}")

# --- data loading -----------------------------------------
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

data_transforms = transforms.Compose([

    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),

    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset_train = ImageFolder(root=train_dir, transform=data_transforms)
dataset_valid = ImageFolder(root=valid_dir, transform=data_transforms)
dataset_test = ImageFolder(root=test_dir, transform=data_transforms)
# only got one label per image so no one-hot encoding

# --- CONFIG -----------------
BATCHSIZE = 64
EPOCHS = 15
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

if dataset_train.classes == dataset_valid.classes == dataset_test.classes:
    print("All datasets are synchronized!")
else:
    print("Warning: Mismatch in class labels between splits.")

print(f"CONFIG ---------")
print(f"Num Classes: {len(dataset_train.classes)}")
print(f"BATCHSIZE: {BATCHSIZE}")
print(f"Saving Checkpoints to: {CHECKPOINT_DIR}")
print(f"--------------")

train_loader = DataLoader(dataset_train, batch_size=BATCHSIZE, shuffle=True, num_workers=4,pin_memory=True)
val_loader = DataLoader(dataset_valid, batch_size=BATCHSIZE, shuffle=False, num_workers=4,pin_memory=True)
test_loader = DataLoader(dataset_test, batch_size=BATCHSIZE, shuffle=False, num_workers=4,pin_memory=True)

with open("cat_to_name.json", "r") as f:
    CAT_TO_NAME = json.load(f)

def objective(trial):
    # --- Load pretrained ResNet50 -----------------------------------
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # freeze all pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    hidden_units = trial.suggest_int("hidden_units", 64, 4096, log=True)
    dropout_rate = trial.suggest_float("dropout", 0.0, 0.5)

    # Replace classifier layer with our own
    num_classes = len(dataset_train.classes)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(p=dropout_rate),
        nn.Linear(hidden_units, num_classes)
    )
    model = model.to(DEVICE)

    # let optuna figure out the parameters
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.fc.parameters(), lr=lr)
    
    best_accuracy = 0.0
    for epoch in range(EPOCHS):
        # Training
        # if training set is unbalanced we might want to change weigt of loss funciton
        train_model(model, train_loader, optimizer, DEVICE)
        
        # Validation
        accuracy, loss = eval_model(model, val_loader, DEVICE)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_params": trial.params,
                "best_value": accuracy,
                "classes": dataset_train.classes,
                "epoch": epoch,
                "cat_to_name": CAT_TO_NAME,      
            }, os.path.join(CHECKPOINT_DIR, f"model_trial_{trial.number}.pth"))

        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy

def callback(study, trial):
    if study.best_trial.number == trial.number:
        # Rename the file of the best trial to "best_model.pth"
        best_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
        trial_path = os.path.join(CHECKPOINT_DIR, f"model_trial_{trial.number}.pth")
        if os.path.exists(trial_path):
            os.replace(trial_path, best_path)


if __name__ == "__main__":
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3", 
        study_name="resnet50_flower_finetuning",
        direction="maximize", 
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=20, callbacks=[callback])

    print("\n--- OPTIMIZATION FINISHED ---")
    print(f"Best value (Accuracy): {study.best_value:.4f}")
    print("Best parameters:", study.best_params)

    print("Done on:", DEVICE)