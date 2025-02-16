import sys
import os
import cv2
import torch
import requests
import numpy as np

# Add project root to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.bmn_action_recognition import BMN_ActionRecognition

# Load trained model
model = BMN_ActionRecognition(num_classes=10)
model.load_state_dict(torch.load("C:\\Users\\Newtons\\Downloads\\ML Engineering Assignment\\action-recognition-via-video-stream\\models\\bmn_action_recognition.py"))
model.eval()

video_url = "http://localhost:8000/video_feed"
cap = cv2.VideoCapture(video_url)

def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))  
    frame = np.array(frame) / 255.0  
    frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)  
    return frame.unsqueeze(0)  

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    processed_frame = preprocess_frame(frame)
    with torch.no_grad():
        action_pred, _, _, _ = model(processed_frame)
        predicted_label = torch.argmax(action_pred).item()

    cv2.putText(frame, f"Action: {predicted_label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Live Action Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
