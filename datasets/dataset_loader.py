import torch
from torch.data import DataLoader, Dataset
import os
import cv2
import numpy as np

class UCF101Dataset(Dataset):
    def __init__(self, data_dir):
        self.video_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.avi')]

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)

        cap.release()
        frames = np.array(frames) / 255.0  
        frames = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2)  
        label = torch.randint(0, 10, (1,))  

        return frames, label, torch.tensor(0.1), torch.tensor(0.9)

def get_dataloader(train=True, batch_size=4):
    dataset = UCF101Dataset(data_dir="data/train" if train else "data/test")
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
