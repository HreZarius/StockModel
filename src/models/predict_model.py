import torch
import torch.nn as nn
import yaml
import pandas as pd
from src.models.train_model import LSTMModel
from src.models.attention_lstm import AttentionLSTM

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def prediction(model, input_data):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
        output = model(input_tensor)
    return output.numpy()

def load_model(model_path, model_class, *args, **kwargs):
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(config):
    df = pd.read_csv(config["processed_data_path"], index_col=config["index_col"])
    window_size = config["train_params"]["window_size"]
    input_data = df.values[-window_size:]
    
    # Choose model type based on config
    if config.get("model_type", "lstm") == "attention":
        model_class = AttentionLSTM
    else:
        model_class = LSTMModel
    
    model = load_model(
        config["model_path"], 
        model_class, 
        input_dim=df.shape[1],
        **config["model_params"]
    )
    
    pred = prediction(model, input_data)
    return pred


if __name__ == "__main__":
    config = load_config()

    import pandas as pd
    df = pd.read_csv(config["processed_data_path"], index_col=config["index_col"])
    input_data = df.values[-60:]  # Last 60 time steps as input
    preds = predict(config, input_data)
    print("Predictions:", preds)