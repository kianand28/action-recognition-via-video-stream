import matplotlib.pyplot as plt
import time
import random
import torch
import cv2
from models.bmn_action_recognition import BMN_ActionRecognition

# Load trained model
model = BMN_ActionRecognition(num_classes=10)
model.load_state_dict(torch.load("models/bmn_model.pth"))
model.eval()

# Dummy list to store detected actions
detected_actions = []
labels = ["Running", "Jumping", "Walking", "Sitting", "Waving", "Dancing", "Falling", "Standing", "Climbing", "Kicking"]

def update_plot():
    plt.clf()
    plt.bar(labels, [detected_actions.count(i) for i in range(10)])
    plt.xlabel("Actions")
    plt.ylabel("Frequency")
    plt.title("Real-time Activity Recognition Analytics")
    plt.pause(0.1)

while True:
    action_pred = torch.randint(0, 10, (1,))  # Simulated inference
    detected_actions.append(action_pred.item())

    update_plot()
    time.sleep(1)
