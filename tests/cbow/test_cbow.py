import unittest
import numpy as np
import os
from cbow.cbow import CBOW
from functions import Functions

class TestCBOW(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_file = "test_mock_cbow.txt"
        with open(cls.test_file, "w", encoding="utf-8") as f:
            f.write("a b c d e") 
        
        cls.context_size = 2
        cls.learning_rate = 0.05
        cls.N = 2

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_file):
            os.remove(cls.test_file)

    def test_forward_pass_math(self):
        # Validates the mathematical accuracy of the hidden layer and output probability distributions
        model = CBOW(self.test_file, self.context_size, self.learning_rate, self.N)
        
        model.W1 = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.5]])
        model.W2 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]])

        context_ids = [0, 2]
        
        expected_u = np.array([0.04, 0.08, 0.12, 0.16, 0.20])
        expected_pred = Functions.softmax(expected_u)

        predictions = model.forward_pass(context_ids, context_size=2)
        np.testing.assert_almost_equal(predictions, expected_pred, decimal=5)

    def test_backpropagation_math(self):
        # Validates the gradient calculations and subsequent weight matrix updates
        model = CBOW(self.test_file, self.context_size, self.learning_rate, self.N)
        
        model.W1 = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.5]])
        model.W2 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]])
        
        model.h = np.array([0.2, 0.2])
        model.y_pred = np.array([0.1, 0.6, 0.1, 0.1, 0.1])
        
        model.backpropagation(output_word_id=1, context_word_ids=[0, 2], context_size=2)
        
        np.testing.assert_almost_equal(model.W1[0], [0.09875, 0.09875], decimal=5)

    def test_train_execution(self):
        # Ensures the full training loop executes without runtime or dimension errors
        model = CBOW(self.test_file, self.context_size, self.learning_rate, self.N)
        
        try:
            model.train(epochs=1, print_interval=10)
            execution_successful = True
        except Exception as e:
            execution_successful = False
            print(f"Train loop failed: {e}")
            
        self.assertTrue(execution_successful)