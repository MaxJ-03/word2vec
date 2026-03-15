import unittest
import numpy as np
import os
from unittest.mock import patch
from cbow.cbow_neg_sample import CBOWNegativeSampling
from functions import Functions

class TestCBOWNegativeSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_file = "test_mock_cbow_ns.txt"
        with open(cls.test_file, "w", encoding="utf-8") as f:
            f.write("a b c d e")
        
        cls.context_size = 2
        cls.learning_rate = 0.05
        cls.N = 2

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_file):
            os.remove(cls.test_file)

    @patch('data_proc.DataProcessing.generate_negative_samples')
    def test_weight_update_math(self, mock_neg_samples):
        # Validates loss calculations and weight updates for the true target and generated negative samples
        mock_neg_samples.return_value = [3, 4]
        
        model = CBOWNegativeSampling(self.test_file, self.context_size, self.learning_rate, self.N, negative_sampling_size=2)
        
        model.W1 = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.5]])
        model.W2 = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.5]])

        f_true = Functions.sigmoid(np.array(0.08)) 
        loss_true = -np.log(f_true + 1e-9)

        f_neg = Functions.sigmoid(np.array([0.16, 0.20]))
        loss_neg = -np.sum(np.log(1 - f_neg + 1e-9))
        
        expected_loss = loss_true + loss_neg
        actual_loss = model.update_weights(output_word_id=1, context_vector_ids=[0, 2], context_size=2)
        
        self.assertAlmostEqual(actual_loss, expected_loss, places=5)

    def test_train_execution(self):
        # Ensures the full training loop executes without runtime or dimension errors
        model = CBOWNegativeSampling(self.test_file, self.context_size, self.learning_rate, self.N, negative_sampling_size=2)
        
        try:
            model.train(epochs=1, print_interval=10)
            execution_successful = True
        except Exception as e:
            execution_successful = False
            print(f"Train loop failed: {e}")
            
        self.assertTrue(execution_successful)