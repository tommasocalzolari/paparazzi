# Bebop Tree Classifier

This project is designed for training fast classifiers suitable for deployment on a Bebop drone using tree patch data. The goal is to create an efficient and effective model that can classify tree patches based on various features extracted from the data.

## Project Structure

```
bebop-tree-classifier
├── src
│   └── bebop_classifier
│       ├── __init__.py
│       ├── data
│       │   ├── dataset.py
│       │   └── transforms.py
│       ├── features
│       │   └── patch_features.py
│       ├── models
│       │   ├── train.py
│       │   ├── predict.py
│       │   └── export.py
│       └── utils
│           └── io.py
├── configs
│   ├── train.yaml
│   └── deploy.yaml
├── scripts
│   ├── train.py
│   └── evaluate.py
├── tests
│   └── test_train.py
├── data
│   ├── raw
│   └── processed
├── models
│   └── .gitkeep
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd bebop-tree-classifier
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model, run the following script:

```bash
python scripts/train.py
```

This will initialize the training process using the `Trainer` class defined in `src/bebop_classifier/models/train.py`.

### Evaluating the Model

After training, you can evaluate the model's performance with:

```bash
python scripts/evaluate.py
```

This script uses the `Predictor` class from `src/bebop_classifier/models/predict.py` to assess the model.

## Configuration

Configuration settings for training and deployment can be found in the `configs` directory:

- `train.yaml`: Contains parameters like learning rate, batch size, and number of epochs for training.
- `deploy.yaml`: Contains deployment parameters and environment settings.

## Data

Raw tree patch data should be placed in the `data/raw` directory, while processed data will be stored in the `data/processed` directory.

## Testing

Unit tests for the training module are located in `tests/test_train.py`. You can run the tests to ensure that the training process works as expected.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.