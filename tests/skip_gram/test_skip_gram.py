import unittest
import numpy as np
import os
from skip_gram.skip_gram import SkipGram
from functions import Functions

class TestSkipGram(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_file = "test_mock_sg.txt"
        with open(cls.test_file, "w", encoding="utf-8") as f:
            f.write("a b c d e") # V = 5
        
        cls.context_size = 2
        cls.learning_rate = 0.05
        cls.N = 2

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_file):
            os.remove(cls.test_file)

    def test_forward_pass_math(self):
        model = SkipGram(self.test_file, self.context_size, self.learning_rate, self.N)
        
        # Fixed Matrices (W1: 5x2, W2: 2x5)
        model.W1 = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.5]])
        model.W2 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]])

        # Input Target: 'b' (1)
        input_id = 1
        

        # h = W1[1] = [0.2, 0.2]
        # u = W2.T @ h = [0.04, 0.08, 0.12, 0.16, 0.20]
        expected_u = np.array([0.04, 0.08, 0.12, 0.16, 0.20])
        expected_pred = Functions.softmax(expected_u)

        predictions = model.forward_pass(input_id)
        np.testing.assert_almost_equal(predictions, expected_pred, decimal=5, err_msg="Standard SkipGram math failed!")