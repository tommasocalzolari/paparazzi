from src.bebop_classifier.data.dataset import Dataset
from src.bebop_classifier.features.patch_features import PatchFeatures
from src.bebop_classifier.models.train import Trainer
from src.bebop_classifier.utils.io import read_config

def main():
    # Load configuration
    config = read_config('configs/train.yaml')

    # Load and preprocess data
    dataset = Dataset(config['data'])
    data = dataset.load_data()
    processed_data = dataset.preprocess(data)

    # Extract features
    feature_extractor = PatchFeatures()
    features = feature_extractor.extract_features(processed_data)

    # Train the model
    trainer = Trainer(config['training'])
    trainer.train_model(features)

    # Save the trained model
    trainer.save_model(config['model']['save_path'])

if __name__ == "__main__":
    main()