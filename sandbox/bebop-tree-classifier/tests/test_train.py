import unittest
from src.bebop_classifier.models.train import Trainer

class TestTrainer(unittest.TestCase):

    def setUp(self):
        self.trainer = Trainer()

    def test_train_model(self):
        # Assuming we have a mock dataset and parameters for training
        mock_data = ...  # Replace with actual mock data
        mock_params = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 10
        }
        model = self.trainer.train_model(mock_data, mock_params)
        self.assertIsNotNone(model)

    def test_save_model(self):
        model = ...  # Replace with an actual model instance
        save_path = 'path/to/save/model'
        result = self.trainer.save_model(model, save_path)
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()