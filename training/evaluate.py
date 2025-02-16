import torch
from models.bmn_action_recognition import BMN_ActionRecognition
from datasets.dataset_loader import get_dataloader

model = BMN_ActionRecognition(num_classes=10)
test_loader = get_dataloader(train=False)

def evaluate(model, dataloader):
    model.eval()
    total_correct, total_samples = 0, 0

    with torch.no_grad():
        for videos, labels, _, _ in dataloader:
            action_preds, _, _, _ = model(videos)
            predicted_labels = torch.argmax(action_preds, dim=1)
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

evaluate(model, test_loader)
