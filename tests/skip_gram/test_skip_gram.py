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
        model = SkipGram(self.test_file, self.context_size, self.learning_rate, self.N)
        
        model.W1 = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.5]])
        model.W2 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]])

        input_vector_id = 1
        
        expected_u = np.array([0.04, 0.08, 0.12, 0.16, 0.20])
        expected_pred = Functions.softmax(expected_u)

        predictions = model.forward_pass(input_vector_id)
        np.testing.assert_almost_equal(predictions, expected_pred, decimal=5)

    def test_backpropagation_math(self):
        # Validates the gradient calculations and subsequent weight matrix updates
        model = SkipGram(self.test_file, self.context_size, self.learning_rate, self.N)
        
        model.W1 = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.5]])
        model.W2 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]])
        
        model.backpropagation(input_vector_id=1, context_vectors_ids=[0, 2], context_size=2)
        
        self.assertNotEqual(model.W1[1][0], 0.2)
        self.assertNotEqual(model.W2[0][0], 0.1)

    def test_numerical_gradient_check(self):
        # Validates analytical gradients against finite-difference estimates
        model = SkipGram(self.test_file, self.context_size, self.learning_rate, self.N)
        model.W1 = np.random.rand(model.V, model.N)
        model.W2 = np.random.rand(model.N, model.V)

        input_vector_id = 1
        context_vectors_ids = [0, 2]
        context_size = 2
        epsilon = 1e-5

        loss, dW2_analytical, dW1_analytical = model.compute_gradients(input_vector_id, context_vectors_ids, context_size)

        dW2_numerical = np.zeros_like(model.W2)
        for i in range(model.W2.shape[0]):
            for j in range(model.W2.shape[1]):
                orig = model.W2[i, j]
                model.W2[i, j] = orig + epsilon
                loss_plus, _, _ = model.compute_gradients(input_vector_id, context_vectors_ids, context_size)
                model.W2[i, j] = orig - epsilon
                loss_minus, _, _ = model.compute_gradients(input_vector_id, context_vectors_ids, context_size)
                dW2_numerical[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
                model.W2[i, j] = orig

        np.testing.assert_allclose(dW2_analytical, dW2_numerical, rtol=1e-4, atol=1e-4)

        dW1_numerical = np.zeros_like(model.W1)
        for j in range(model.W1.shape[1]):
            orig = model.W1[input_vector_id, j]
            model.W1[input_vector_id, j] = orig + epsilon
            loss_plus, _, _ = model.compute_gradients(input_vector_id, context_vectors_ids, context_size)
            model.W1[input_vector_id, j] = orig - epsilon
            loss_minus, _, _ = model.compute_gradients(input_vector_id, context_vectors_ids, context_size)
            dW1_numerical[input_vector_id, j] = (loss_plus - loss_minus) / (2 * epsilon)
            model.W1[input_vector_id, j] = orig

        np.testing.assert_allclose(dW1_analytical, dW1_numerical[input_vector_id], rtol=1e-4, atol=1e-4)

    def test_train_execution(self):
        # Ensures the full training loop executes without runtime or dimension errors
        model = SkipGram(self.test_file, self.context_size, self.learning_rate, self.N)
        
        try:
            model.train(epochs=1, print_interval=10)
            execution_successful = True
        except Exception as e:
            execution_successful = False
            print(f"Train loop failed: {e}")
            
        self.assertTrue(execution_successful)

    def test_train_single_token_dataset(self):
        # Ensures training remains numerically stable when context is empty
        solo_file = "test_mock_sg_single_token.txt"
        try:
            with open(solo_file, "w", encoding="utf-8") as f:
                f.write("solo")

            model = SkipGram(solo_file, context_size=4, learning_rate=self.learning_rate, hidden_layer_size=self.N)
            model.train(epochs=1, print_interval=10)

            self.assertFalse(np.isnan(model.W1).any())
            self.assertFalse(np.isnan(model.W2).any())
        finally:
            if os.path.exists(solo_file):
                os.remove(solo_file)