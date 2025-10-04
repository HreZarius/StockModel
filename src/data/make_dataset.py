import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import torch
from src.features.data_augmentation import augment_sequence

def load_data(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    df = pd.read_csv(config['data_path'], index_col=config['index_col'])
    return df

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    
    return np.array(X), np.array(y)

def scale_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.values)
    return scaled_data, scaler

class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size=60, transform=None, augment=False):
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        self.X, self.y = create_sequences(data, window_size)
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx].copy()
        y = self.y[idx].copy()
        
        # Apply augmentation during training
        if self.augment:
            X = augment_sequence(X)
        
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        if self.transform:
            X, y = self.transform(X, y)
        return X, y

if __name__ == "__main__":
    df = load_data()
    dataset = TimeSeriesDataset(df, window_size=60)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for X_batch, y_batch in dataloader:
        print("X_batch shape:", X_batch.shape)  
        print("y_batch shape:", y_batch.shape)  
        break


    