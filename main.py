import argparse
from src.models.train_model import load_config, train_model
from src.models.predict_model import predict
from src.features.make_features import make_features

def main():
    parser = argparse.ArgumentParser(description="StockModel runner")
    parser.add_argument("--mode", choices=["train", "predict"], required=True, help="Mode: train or predict")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode == "train":
        make_features(config["raw_data_path"], config["processed_data_path"])
        train_model(config)
    elif args.mode == "predict":
        predict(config)

if __name__ == "__main__":
    main()