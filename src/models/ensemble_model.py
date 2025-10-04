import torch
import torch.nn as nn
import numpy as np
from src.models.train_model import LSTMModel

class EnsembleModel:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights else [1.0/len(models)] * len(models)
    
    def predict(self, x):
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred.numpy())
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        return ensemble_pred

def create_ensemble_models(config, num_models=3):
    """Создает ансамбль из нескольких моделей с разными параметрами"""
    models = []
    
    # Разные архитектуры для разнообразия
    model_configs = [
        {'hidden_dim': 64, 'num_layers': 2, 'dropout': 0.1},
        {'hidden_dim': 128, 'num_layers': 3, 'dropout': 0.2},
        {'hidden_dim': 256, 'num_layers': 2, 'dropout': 0.3}
    ]
    
    for i, model_config in enumerate(model_configs):
        model = LSTMModel(
            input_dim=config.get('input_dim', 25),
            output_dim=config.get('output_dim', 1),
            **model_config
        )
        models.append(model)
    
    return models
