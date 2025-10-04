import pandas as pd
import numpy as np
import yaml
import os
from sklearn.preprocessing import StandardScaler


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def add_sma(df, column="Close", window=14):
    df[f"SMA_{window}"] = df[column].rolling(window).mean()
    return df


def add_ema(df, column="Close", window=14):
    df[f"EMA_{window}"] = df[column].ewm(span=window, adjust=False).mean()
    return df


def add_rsi(df, column="Close", window=14):
    delta = df[column].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=window, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(span=window, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    df[f"RSI_{window}"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df, column="Close", fast=5, slow=10, signal=3):
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    return df


def add_volatility(df, column="Close", window=14):
    df[f"Volatility_{window}"] = df[column].pct_change().rolling(window).std()
    return df


def build_features(df, config):
    f = df.copy()
    
    # Basic features
    f['Price_Change'] = f['Close'].pct_change()
    f['High_Low_Pct'] = (f['High'] - f['Low']) / f['Close']
    f['Open_Close_Pct'] = (f['Close'] - f['Open']) / f['Open']
    f['Volume_Change'] = f['Volume'].pct_change()
    
    # Moving averages
    f['SMA_5'] = f['Close'].rolling(5).mean()
    f['SMA_10'] = f['Close'].rolling(10).mean()
    f['SMA_20'] = f['Close'].rolling(20).mean()
    f['EMA_5'] = f['Close'].ewm(span=5, adjust=False).mean()
    f['EMA_10'] = f['Close'].ewm(span=10, adjust=False).mean()
    
    # Technical indicators
    f['RSI_14'] = add_rsi_simple(f['Close'], 14)
    f['MACD'] = add_macd_simple(f['Close'])
    f['Bollinger_Upper'] = f['Close'].rolling(20).mean() + 2 * f['Close'].rolling(20).std()
    f['Bollinger_Lower'] = f['Close'].rolling(20).mean() - 2 * f['Close'].rolling(20).std()
    f['Bollinger_Position'] = (f['Close'] - f['Bollinger_Lower']) / (f['Bollinger_Upper'] - f['Bollinger_Lower'])
    
    # Volatility and momentum
    f['Volatility_5'] = f['Close'].pct_change().rolling(5).std()
    f['Volatility_10'] = f['Close'].pct_change().rolling(10).std()
    f['Momentum_5'] = f['Close'] / f['Close'].shift(5) - 1
    f['Momentum_10'] = f['Close'] / f['Close'].shift(10) - 1
    
    # Volume indicators
    f['Volume_SMA'] = f['Volume'].rolling(10).mean()
    f['Volume_Ratio'] = f['Volume'] / f['Volume_SMA']
    
    f = f.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(f) == 0:
        raise ValueError("No data left after removing NaN values.")
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(f)
    return pd.DataFrame(scaled, index=f.index, columns=f.columns)

def add_rsi_simple(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_macd_simple(series, fast=12, slow=26):
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    return ema_fast - ema_slow

def make_features(raw_data_path, processed_data_path):
    df = pd.read_csv(raw_data_path, index_col="Date")
    features = build_features(df, {})
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    features.to_csv(processed_data_path)
    return features

if __name__ == "__main__":
    config = load_config()
    df = pd.read_csv(config["interim_data_path"], index_col=config["index_col"])
    features = make_features(df, config)
    os.makedirs(os.path.dirname(config["processed_data_path"]), exist_ok=True)
    features.to_csv(config["processed_data_path"])
    print(f"Features saved to {config['processed_data_path']}")
    print(features.head())
