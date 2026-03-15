import unittest
import numpy as np
from functions import Functions

class TestFunctions(unittest.TestCase):
    def test_sigmoid(self):
        # Verifies standard sigmoid output for zero.
        self.assertEqual(Functions.sigmoid(0), 0.5)
        
        # Verifies numerical stability clipping limits to prevent overflow.
        self.assertAlmostEqual(Functions.sigmoid(100), 1.0, places=5)
        self.assertAlmostEqual(Functions.sigmoid(-100), 0.0, places=5)

    def test_softmax(self):
        # Verifies softmax probabilities sum to exactly 1.0.
        x = np.array([1.0, 2.0, 3.0])
        result = Functions.softmax(x)
        self.assertAlmostEqual(np.sum(result), 1.0, places=5)
        
        # Verifies shift-invariance optimization for large numbers.
        x_large = np.array([1000.0, 1001.0, 1002.0])
        result_large = Functions.softmax(x_large)
        np.testing.assert_almost_equal(result, result_large, decimal=5)