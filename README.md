# StockModel

A deep learning model for stock price prediction using LSTM with attention mechanism and data augmentation.

## Features

- **LSTM with Attention**: Advanced neural network architecture for time series prediction
- **Technical Indicators**: RSI, MACD, Bollinger Bands, moving averages, and more
- **Data Augmentation**: Noise injection, time shifting, and scaling for better generalization
- **Ensemble Support**: Multiple model architectures for improved predictions
- **Early Stopping**: Prevents overfitting with automatic training termination
- **Configurable**: Easy parameter tuning through YAML configuration

## Project Structure

```
StockModel/
├── src/
│   ├── data/
│   │   └── make_dataset.py          # Data loading and preprocessing
│   ├── features/
│   │   ├── make_features.py        # Feature engineering
│   │   └── data_augmentation.py    # Data augmentation techniques
│   └── models/
│       ├── train_model.py          # LSTM model training
│       ├── attention_lstm.py       # Attention LSTM implementation
│       ├── predict_model.py        # Model prediction
│       └── ensemble_model.py       # Ensemble methods
├── data/
│   ├── dataset.csv                 # Raw stock data
│   ├── processed/                  # Processed features
│   └── interim/                    # Intermediate data
├── models/                         # Trained model files
├── config.yaml                     # Configuration file
└── main.py                        # Main entry point
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd StockModel
```

2. Install dependencies:
```bash
pip install torch pandas numpy scikit-learn pyyaml
```

## Usage

### Training

Train a new model with default settings:
```bash
python main.py --mode train --config config.yaml
```

### Prediction

Make predictions using trained model:
```bash
python main.py --mode predict --config config.yaml
```

## Configuration

Edit `config.yaml` to customize model parameters:

```yaml
# Data paths
data_path: "data/dataset.csv"
processed_data_path: "data/processed/stock_features.csv"
model_path: "models/best_model.pth"

# Model configuration
model_type: "attention"  # "lstm" or "attention"
model_params:
  hidden_dim: 128
  num_layers: 3
  output_dim: 1
  dropout: 0.2

# Training parameters
train_params:
  window_size: 60
  batch_size: 64
  epochs: 200
  lr: 0.0005
  val_split: 0.2
```

## Model Architecture

### LSTM with Attention
- **Input**: Time series features (25+ technical indicators)
- **LSTM Layers**: 3 layers with 128 hidden units
- **Attention Mechanism**: Focuses on important time steps
- **Output**: Single value prediction
- **Regularization**: Dropout (0.2) and weight decay

### Features
- Price changes and ratios
- Moving averages (SMA, EMA)
- Technical indicators (RSI, MACD, Bollinger Bands)
- Volume indicators
- Momentum and volatility measures

## Data Augmentation

The model includes several data augmentation techniques:
- **Noise Injection**: Adds small random noise
- **Time Shifting**: Shifts sequences in time
- **Scaling**: Random scaling of features

## Performance

The model achieves:
- **Validation Loss**: ~0.83 (with attention mechanism)
- **Training Stability**: Early stopping prevents overfitting
- **Feature Rich**: 25+ technical indicators for comprehensive analysis

## Advanced Features

### Ensemble Models
Support for combining multiple model architectures:
- Different LSTM configurations
- Weighted predictions
- Improved generalization

### Attention Mechanism
- Focuses on relevant time steps
- Better long-term dependency modeling
- Improved prediction accuracy

## Requirements

- Python 3.7+
- PyTorch
- pandas
- numpy
- scikit-learn
- PyYAML

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- Technical indicators implementation
- LSTM architecture design
- Attention mechanism implementation
