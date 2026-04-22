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

    def test_numerical_gradient_check(self):
        # Validates analytical gradients against finite-difference estimates
        model = SkipGramHierarchical(self.test_file, self.context_size, self.learning_rate, self.N)
        model.W1 = np.random.rand(model.V, model.N)
        model.W2 = np.random.rand(model.V - 1, model.N)

        input_vector_id = 1
        context_words = ['a', 'c']
        mock_huffman_dict = {
            'a': {'code': [1], 'path': [0]},
            'c': {'code': [-1], 'path': [1]}
        }
        epsilon = 1e-5

        loss, dW2_analytical, dW1_analytical, flat_paths = model.compute_gradients(input_vector_id, context_words, mock_huffman_dict)

        dW2_full_analytical = np.zeros_like(model.W2)
        np.add.at(dW2_full_analytical, flat_paths, dW2_analytical)

        dW2_numerical = np.zeros_like(model.W2)
        unique_paths = np.unique(flat_paths)
        for p in unique_paths:
            for j in range(model.W2.shape[1]):
                orig = model.W2[p, j]
                model.W2[p, j] = orig + epsilon
                loss_plus, _, _, _ = model.compute_gradients(input_vector_id, context_words, mock_huffman_dict)
                model.W2[p, j] = orig - epsilon
                loss_minus, _, _, _ = model.compute_gradients(input_vector_id, context_words, mock_huffman_dict)
                dW2_numerical[p, j] = (loss_plus - loss_minus) / (2 * epsilon)
                model.W2[p, j] = orig

        for p in unique_paths:
            np.testing.assert_allclose(dW2_full_analytical[p], dW2_numerical[p], rtol=1e-4, atol=1e-4)

        dW1_numerical = np.zeros_like(model.W1)
        for j in range(model.W1.shape[1]):
            orig = model.W1[input_vector_id, j]
            model.W1[input_vector_id, j] = orig + epsilon
            loss_plus, _, _, _ = model.compute_gradients(input_vector_id, context_words, mock_huffman_dict)
            model.W1[input_vector_id, j] = orig - epsilon
            loss_minus, _, _, _ = model.compute_gradients(input_vector_id, context_words, mock_huffman_dict)
            dW1_numerical[input_vector_id, j] = (loss_plus - loss_minus) / (2 * epsilon)
            model.W1[input_vector_id, j] = orig

        np.testing.assert_allclose(dW1_analytical, dW1_numerical[input_vector_id], rtol=1e-4, atol=1e-4)

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