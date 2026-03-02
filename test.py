import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import load_model

def evaluate_test_set(model, test_loader, classes, cat_to_name, device):
    model.eval()
    results = []
    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            probs = torch.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probs, k=5, dim=1)

            for i in range(len(targets)):
                true_class = classes[targets[i].item()]
                true_label = cat_to_name[true_class]
                
                pred_classes = [classes[idx] for idx in top_indices[i].tolist()]
                pred_labels = [cat_to_name[c] for c in pred_classes]
                pred_probs = top_probs[i].tolist()

                is_correct = true_class == pred_classes[0]
                correct += is_correct
                total += 1

                results.append({
                    "true_label": true_label,
                    "predicted": list(zip(pred_labels, pred_probs)),
                    "correct": is_correct
                })

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f} ({correct}/{total})")
    return results, accuracy

DEVICE = torch.device("cpu")

data_dir = 'flowers'
test_dir = data_dir + '/test'

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset_test = ImageFolder(root=test_dir, transform=data_transforms)

# --- CONFIG -----------------
BATCHSIZE = 32
test_loader = DataLoader(dataset_test, batch_size=BATCHSIZE, shuffle=False)

if __name__ == "__main__":
    checkpoint_path = "checkpoints/best_model.pth"

    load_model.load_model(checkpoint_path)

    results, accuracy = evaluate_test_set(
        load_model.model, test_loader, load_model.classes, load_model.cat_to_name, DEVICE
    )

    wrong = [r for r in results if not r["correct"]]
    print(f"\nWrong predictions: {len(wrong)}")
    for r in wrong[:5]:  # show first 5
        print(f"True: {r['true_label']} → Predicted: {r['predicted'][0]}")