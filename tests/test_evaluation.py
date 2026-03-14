import unittest
import numpy as np
import io
from unittest.mock import patch, MagicMock
from evaluation import get_similar_words

class TestEvaluation(unittest.TestCase):
    def setUp(self):
        # Create a "fake" model so we don't have to initialize a real neural network
        self.mock_model = MagicMock()
        self.mock_model.data_processing.vocabulary = ['apple', 'banana', 'car', 'dog']
        self.mock_model.data_processing.word_to_id = {'apple': 0, 'banana': 1, 'car': 2, 'dog': 3}
        self.mock_model.data_processing.id_to_word = {0: 'apple', 1: 'banana', 2: 'car', 3: 'dog'}

        # Hardcode W1 with highly predictable vectors
        self.mock_model.W1 = np.array([
            [1.0, 0.0],  # apple
            [1.0, 0.0],  # banana (Identical to apple, Cosine Sim = 1.0)
            [0.0, 1.0],  # car    (Orthogonal to apple, Cosine Sim = 0.0)
            [-1.0, 0.0]  # dog    (Opposite of apple, Cosine Sim = -1.0)
        ])

    # @patch intercepts 'sys.stdout' so we can capture the print() outputs
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_get_similar_words_valid(self, mock_stdout):
        # We ask for the word most similar to 'apple'
        get_similar_words(self.mock_model, 'apple', top_n=1)
        
        # Grab the text that was printed to the terminal
        printed_output = mock_stdout.getvalue()

        # Check if the math and output are correct
        self.assertIn("Words most similar to 'apple':", printed_output)
        self.assertIn("banana", printed_output, "'banana' should be the closest word")
        self.assertIn("(Score: 1.0000)", printed_output, "Identical vectors must have a score of 1.0000")

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_get_similar_words_out_of_vocab(self, mock_stdout):
        # Test a word that doesn't exist
        get_similar_words(self.mock_model, 'spaceship')
        
        printed_output = mock_stdout.getvalue()
        self.assertIn("Word 'spaceship' not found in vocabulary.", printed_output)