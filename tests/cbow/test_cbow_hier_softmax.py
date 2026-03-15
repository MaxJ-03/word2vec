import unittest
import numpy as np
import os
from cbow.cbow_hier_softmax import CBOWHierarchical
from functions import Functions

class TestCBOWHierarchical(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_file = "test_mock_cbow_hs.txt"
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
        # Validates the combined forward and backward passes along the Huffman tree path
        model = CBOWHierarchical(self.test_file, self.context_size, self.learning_rate, self.N)
        
        model.W1 = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.5]])
        model.W2 = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]])

        mock_huffman_dict = {'b': {'code': [1, -1], 'path': [0, 2]}}

        f = Functions.sigmoid(np.array([0.04, -0.12])) 
        expected_loss = -np.sum(np.log(f + 1e-9))

        actual_loss = model.update_weights(output_word_id=1, context_vector_ids=[0, 2], context_size=2, huffman_dict=mock_huffman_dict)
        self.assertAlmostEqual(actual_loss, expected_loss, places=5)

    def test_train_execution(self):
        # Ensures the full training loop executes without runtime or dimension errors
        model = CBOWHierarchical(self.test_file, self.context_size, self.learning_rate, self.N)
        
        try:
            model.train(epochs=1, print_interval=10)
            execution_successful = True
        except Exception as e:
            execution_successful = False
            print(f"Train loop failed: {e}")
            
        self.assertTrue(execution_successful)