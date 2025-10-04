import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pandas as pd
import yaml
from src.data.make_dataset import TimeSeriesDataset
from src.models.attention_lstm import AttentionLSTM


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, output_dim=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc3 = nn.Linear(hidden_dim // 4, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        return out


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def train_model(config):
    df = pd.read_csv(config["processed_data_path"], index_col=config["index_col"])
    dataset = TimeSeriesDataset(df, window_size=config["train_params"]["window_size"], augment=True)
    
    val_split = config["train_params"].get("val_split", 0.2)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config["train_params"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["train_params"]["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Choose model type
    if config.get("model_type", "lstm") == "attention":
        model = AttentionLSTM(input_dim=df.shape[1], **config["model_params"]).to(device)
    else:
        model = LSTMModel(input_dim=df.shape[1], **config["model_params"]).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train_params"]["lr"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_loss = float("inf")
    patience_counter = 0
    patience = 20
    
    for epoch in range(config["train_params"]["epochs"]):
        model.train()
        train_losses = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                output = model(X)
                val_losses.append(criterion(output, y).item())
        val_loss = sum(val_losses) / len(val_losses)
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), config["model_path"])
            patience_counter = 0
            print(f"Epoch {epoch+1}: val_loss={val_loss:.6f} (saved)")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}: val_loss={val_loss:.6f}")
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

if __name__ == "__main__":
    config = load_config()
    train_model(config)