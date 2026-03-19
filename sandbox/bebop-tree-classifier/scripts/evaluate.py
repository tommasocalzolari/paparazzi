import argparse
import os
from src.bebop_classifier.models.predict import Predictor
from src.bebop_classifier.utils.io import read_config

def evaluate_model(config_path):
    config = read_config(config_path)
    model_path = config['model_path']
    test_data_path = config['test_data_path']

    predictor = Predictor()
    predictor.load_model(model_path)

    results = predictor.predict(test_data_path)
    print("Evaluation Results:", results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the evaluation config file.')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    evaluate_model(args.config)