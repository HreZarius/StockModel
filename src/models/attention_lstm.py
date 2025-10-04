import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, output_dim=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.attention = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc3 = nn.Linear(hidden_dim // 4, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # LSTM output
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_dim]
        
        # Attention mechanism
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)  # [batch, seq_len, 1]
        attended_output = torch.sum(lstm_out * attention_weights, dim=1)  # [batch, hidden_dim]
        
        # Fully connected layers
        out = self.dropout(attended_output)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out
