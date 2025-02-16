import torch
import torch.nn as nn
import torchvision.models as models

class BMN_ActionRecognition(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=256, num_classes=10):
        super(BMN_ActionRecognition, self).__init__()

        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  
        
        self.start_predictor = nn.Linear(feature_dim, 1)
        self.end_predictor = nn.Linear(feature_dim, 1)
        self.confidence_predictor = nn.Linear(feature_dim, 1)

        self.gru = nn.GRU(input_size=feature_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape  
        features = torch.zeros((batch_size, seq_len, 512)).to(x.device)
        
        for t in range(seq_len):
            features[:, t, :] = self.feature_extractor(x[:, t])

        start_scores = self.start_predictor(features)
        end_scores = self.end_predictor(features)
        confidence_scores = self.confidence_predictor(features)

        _, hidden = self.gru(features)
        action_preds = self.fc(hidden[-1])  

        return action_preds, start_scores, end_scores, confidence_scores
