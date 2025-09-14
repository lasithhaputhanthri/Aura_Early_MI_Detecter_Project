import torch
import torch.nn as nn

class CNN_BiLSTM_Classifier(nn.Module):
    def __init__(self, dropout=0.3, hidden_size=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            64, hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, T, 1] -> [B, 1, T]
        x = self.cnn(x)         # [B, 64, T]
        x = x.permute(0, 2, 1)  # [B, T, 64]
        x, _ = self.lstm(x)     # [B, T, 2*H]
        x = x[:, -1, :]         # take last time step
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

