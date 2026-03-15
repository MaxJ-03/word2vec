import unittest
import os
from word2vec_base import Word2VecBase

class TestWord2VecBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_file = "test_mock_base.txt"
        with open(cls.test_file, "w", encoding="utf-8") as f:
            f.write("a b c d e")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_file):
            os.remove(cls.test_file)

    def test_learning_rate_decay(self):
        # Verifies exponential learning rate decay across epochs.
        model = Word2VecBase(self.test_file, context_size=2, learning_rate=0.05, hidden_layer_size=10)
        
        total_epochs = 50
        
        # Simulates epoch 0 to verify the initial learning rate holds.
        model.update_learning_rate(current_epoch=0, total_epochs=total_epochs)
        self.assertEqual(model.learning_rate, 0.05)
        
        # Simulates the final epoch to ensure it floors correctly based on the decay factor.
        model.update_learning_rate(current_epoch=total_epochs, total_epochs=total_epochs)
        self.assertAlmostEqual(model.learning_rate, 0.0005, places=5)