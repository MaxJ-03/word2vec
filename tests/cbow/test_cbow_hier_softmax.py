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

    def test_numerical_gradient_check(self):
        # Validates analytical gradients against finite-difference estimates
        model = CBOWHierarchical(self.test_file, self.context_size, self.learning_rate, self.N)
        model.W1 = np.random.rand(model.V, model.N)
        model.W2 = np.random.rand(model.V - 1, model.N)

        output_word_id = 1
        context_vector_ids = [0, 2]
        context_size = 2
        mock_huffman_dict = {'b': {'code': [1, -1], 'path': [0, 2]}}
        epsilon = 1e-5

        loss, dW2_analytical, dW1_analytical, path = model.compute_gradients(output_word_id, context_vector_ids, context_size, mock_huffman_dict)

        dW2_numerical = np.zeros_like(dW2_analytical)
        for idx, p in enumerate(path):
            for j in range(model.W2.shape[1]):
                orig = model.W2[p, j]
                model.W2[p, j] = orig + epsilon
                loss_plus, _, _, _ = model.compute_gradients(output_word_id, context_vector_ids, context_size, mock_huffman_dict)
                model.W2[p, j] = orig - epsilon
                loss_minus, _, _, _ = model.compute_gradients(output_word_id, context_vector_ids, context_size, mock_huffman_dict)
                dW2_numerical[idx, j] = (loss_plus - loss_minus) / (2 * epsilon)
                model.W2[p, j] = orig

        np.testing.assert_allclose(dW2_analytical, dW2_numerical, rtol=1e-4, atol=1e-4)

        dW1_numerical = np.zeros_like(model.W1)
        for c in context_vector_ids:
            for j in range(model.W1.shape[1]):
                orig = model.W1[c, j]
                model.W1[c, j] = orig + epsilon
                loss_plus, _, _, _ = model.compute_gradients(output_word_id, context_vector_ids, context_size, mock_huffman_dict)
                model.W1[c, j] = orig - epsilon
                loss_minus, _, _, _ = model.compute_gradients(output_word_id, context_vector_ids, context_size, mock_huffman_dict)
                dW1_numerical[c, j] = (loss_plus - loss_minus) / (2 * epsilon)
                model.W1[c, j] = orig
            np.testing.assert_allclose(dW1_analytical, dW1_numerical[c], rtol=1e-4, atol=1e-4)

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