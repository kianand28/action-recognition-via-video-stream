import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.bmn_action_recognition import BMN_ActionRecognition
from datasets.dataset_loader import get_dataloader

model = BMN_ActionRecognition(num_classes=10)
criterion_classification = nn.CrossEntropyLoss()
criterion_boundary = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_loader = get_dataloader(train=True)

for epoch in range(10):
    for videos, labels, start_times, end_times in train_loader:
        optimizer.zero_grad()
        action_preds, start_preds, end_preds, _ = model(videos)

        loss_class = criterion_classification(action_preds, labels)
        loss_start = criterion_boundary(start_preds, start_times)
        loss_end = criterion_boundary(end_preds, end_times)

        total_loss = loss_class + loss_start + loss_end
        total_loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {total_loss.item():.4f}")

# Save trained model
torch.save(model.state_dict(), "models/saved_model.pth")