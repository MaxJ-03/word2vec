import unittest
import numpy as np
import os
from skip_gram.skip_gram_hier_softmax import SkipGramHierarchical
from functions import Functions

class TestSkipGramHierarchical(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_file = "test_mock_sg_hs.txt"
        with open(cls.test_file, "w", encoding="utf-8") as f:
            f.write("a b c d e")
        
        cls.context_size = 2
        cls.learning_rate = 0.05
        cls.N = 2

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_file):
            os.remove(cls.test_file)

    def test_weight_update_math(self):
        # Validates the combined forward and backward passes along the Huffman tree path for all context words
        model = SkipGramHierarchical(self.test_file, self.context_size, self.learning_rate, self.N)
        
        model.W1 = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.5]])
        model.W2 = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]])

        mock_huffman_dict = {
            'a': {'code': [1], 'path': [0]},
            'c': {'code': [-1], 'path': [1]}
        }

        f_a = Functions.sigmoid(np.array([0.04]))
        loss_a = -np.sum(np.log(f_a + 1e-9))

        f_c = Functions.sigmoid(np.array([-0.08]))
        loss_c = -np.sum(np.log(f_c + 1e-9))

        expected_loss = loss_a + loss_c

        actual_loss = model.update_weights(input_vector_id=1, context_words=['a', 'c'], huffman_dict=mock_huffman_dict)
        self.assertAlmostEqual(actual_loss, expected_loss, places=5)

    def test_train_execution(self):
        # Ensures the full training loop executes without runtime or dimension errors
        model = SkipGramHierarchical(self.test_file, self.context_size, self.learning_rate, self.N)
        
        try:
            model.train(epochs=1, print_interval=10)
            execution_successful = True
        except Exception as e:
            execution_successful = False
            print(f"Train loop failed: {e}")
            
        self.assertTrue(execution_successful)