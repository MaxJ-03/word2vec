import unittest
import numpy as np
import os
from unittest.mock import patch
from skip_gram.skip_gram_neg_sample import SkipGramNegativeSampling
from functions import Functions

class TestSkipGramNegativeSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_file = "test_mock_sg_ns.txt"
        with open(cls.test_file, "w", encoding="utf-8") as f:
            f.write("a b c d e")
        
        cls.context_size = 2
        cls.learning_rate = 0.05
        cls.N = 2

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_file):
            os.remove(cls.test_file)

    @patch('data_proc.DataProcessing.generate_negative_samples_list')
    def test_weight_update_math(self, mock_neg_samples_list):
        # Validates loss calculations and weight updates for the true context words and generated negative samples
        mock_neg_samples_list.return_value = [3, 4]

        model = SkipGramNegativeSampling(self.test_file, self.context_size, self.learning_rate, self.N, negative_sampling_size=2)
        
        model.W1 = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.5]])
        model.W2 = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.5]])

        f_true = Functions.sigmoid(np.array([0.04, 0.12]))
        loss_true = -np.sum(np.log(f_true + 1e-9))

        f_neg = Functions.sigmoid(np.array([0.16, 0.20]))
        loss_neg = -np.sum(np.log(1 - f_neg + 1e-9))

        expected_loss = loss_true + loss_neg
        actual_loss = model.update_weights(input_vector_id=1, context_vector_ids=[0, 2])
        
        self.assertAlmostEqual(actual_loss, expected_loss, places=5)

    @patch('data_proc.DataProcessing.generate_negative_samples_list')
    def test_numerical_gradient_check(self, mock_neg_samples_list):
        # Validates analytical gradients against finite-difference estimates
        mock_neg_samples_list.return_value = [3, 4]
        model = SkipGramNegativeSampling(self.test_file, self.context_size, self.learning_rate, self.N, negative_sampling_size=2)
        model.W1 = np.random.rand(model.V, model.N)
        model.W2 = np.random.rand(model.V, model.N)

        input_vector_id = 1
        context_vector_ids = [0, 2]
        epsilon = 1e-5

        loss, dW2_true_analytical, dW2_neg_analytical, dW1_analytical, negative_ids = model.compute_gradients(input_vector_id, context_vector_ids)

        dW2_true_numerical = np.zeros_like(dW2_true_analytical)
        for idx, c_id in enumerate(context_vector_ids):
            for j in range(model.W2.shape[1]):
                orig = model.W2[c_id, j]
                model.W2[c_id, j] = orig + epsilon
                loss_plus, _, _, _, _ = model.compute_gradients(input_vector_id, context_vector_ids)
                model.W2[c_id, j] = orig - epsilon
                loss_minus, _, _, _, _ = model.compute_gradients(input_vector_id, context_vector_ids)
                dW2_true_numerical[idx, j] = (loss_plus - loss_minus) / (2 * epsilon)
                model.W2[c_id, j] = orig

        np.testing.assert_allclose(dW2_true_analytical, dW2_true_numerical, rtol=1e-4, atol=1e-4)

        dW2_neg_numerical = np.zeros_like(dW2_neg_analytical)
        for idx, n_id in enumerate(negative_ids):
            for j in range(model.W2.shape[1]):
                orig = model.W2[n_id, j]
                model.W2[n_id, j] = orig + epsilon
                loss_plus, _, _, _, _ = model.compute_gradients(input_vector_id, context_vector_ids)
                model.W2[n_id, j] = orig - epsilon
                loss_minus, _, _, _, _ = model.compute_gradients(input_vector_id, context_vector_ids)
                dW2_neg_numerical[idx, j] = (loss_plus - loss_minus) / (2 * epsilon)
                model.W2[n_id, j] = orig

        np.testing.assert_allclose(dW2_neg_analytical, dW2_neg_numerical, rtol=1e-4, atol=1e-4)

        dW1_numerical = np.zeros_like(model.W1)
        for j in range(model.W1.shape[1]):
            orig = model.W1[input_vector_id, j]
            model.W1[input_vector_id, j] = orig + epsilon
            loss_plus, _, _, _, _ = model.compute_gradients(input_vector_id, context_vector_ids)
            model.W1[input_vector_id, j] = orig - epsilon
            loss_minus, _, _, _, _ = model.compute_gradients(input_vector_id, context_vector_ids)
            dW1_numerical[input_vector_id, j] = (loss_plus - loss_minus) / (2 * epsilon)
            model.W1[input_vector_id, j] = orig

        np.testing.assert_allclose(dW1_analytical, dW1_numerical[input_vector_id], rtol=1e-4, atol=1e-4)

    def test_train_execution(self):
        # Ensures the full training loop executes without runtime or dimension errors
        model = SkipGramNegativeSampling(self.test_file, self.context_size, self.learning_rate, self.N, negative_sampling_size=2)
        
        try:
            model.train(epochs=1, print_interval=10)
            execution_successful = True
        except Exception as e:
            execution_successful = False
            print(f"Train loop failed: {e}")
            
        self.assertTrue(execution_successful)