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
            f.write("a b c d e") # V = 5
        
        cls.context_size = 2
        cls.learning_rate = 0.05
        cls.N = 2

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_file):
            os.remove(cls.test_file)

    def test_forward_pass_math(self):
        model = CBOW(self.test_file, self.context_size, self.learning_rate, self.N)
        
        # Fixed Matrices (W1: 5x2, W2: 2x5)
        model.W1 = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.5]])
        model.W2 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]])

        # Context: 'a' (0) and 'c' (2)
        context_ids = [0, 2]
        
        # h = ([0.1, 0.1] + [0.3, 0.3]) / 2 = [0.2, 0.2]
        # u = W2.T @ h = [0.04, 0.08, 0.12, 0.16, 0.20]
        expected_u = np.array([0.04, 0.08, 0.12, 0.16, 0.20])
        expected_pred = Functions.softmax(expected_u)

        predictions = model.forward_pass(context_ids, context_size=2)
        np.testing.assert_almost_equal(predictions, expected_pred, decimal=5, err_msg="Standard CBOW math failed!")