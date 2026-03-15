import unittest
import numpy as np
import os
from data_proc import DataProcessing

class TestDataProcessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_file = "test_mock_corpus.txt"
        with open(cls.test_file, "w", encoding="utf-8") as f:
            f.write("apple banana apple cherry dog")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_file):
            os.remove(cls.test_file)

    def setUp(self):
        self.dp = DataProcessing()
        self.dp.load(self.test_file)

    def test_vocabulary_creation(self):
        # Verifies correct tokenization and unique vocabulary sizing.
        self.assertEqual(self.dp.data_length, 5)
        self.assertEqual(len(self.dp.vocabulary), 4)
        self.assertEqual(self.dp.vocabulary_frequency['apple'], 2)

    def test_word_id_mapping(self):
        # Verifies bidirectional mapping integrity between words and integer IDs.
        word = 'cherry'
        word_id = self.dp.word_to_id[word]
        self.assertEqual(self.dp.id_to_word[word_id], word)

    def test_context_window_extraction(self):
        # Verifies context window bounds handling at the beginning of the dataset.
        # Target: 'apple' at index 0. Context size: 2. Expected context: 'banana' (index 1).
        context_ids, actual_size = self.dp.one_hot_encoding_context_ids(id=0, context_size=2)
        self.assertEqual(actual_size, 1)
        
        # Verifies context window bounds handling in the middle of the dataset.
        # Target: 'apple' at index 2. Context size: 2. Expected context: 'banana' (1), 'cherry' (3).
        context_ids_mid, actual_size_mid = self.dp.one_hot_encoding_context_ids(id=2, context_size=2)
        self.assertEqual(actual_size_mid, 2)
        
    def test_negative_sample_generation(self):
        # Verifies the negative sampling generator returns the exact requested number of samples.
        target_id = self.dp.word_to_id['apple']
        samples = self.dp.generate_negative_samples(target_id, num_samples=3)
        self.assertEqual(len(samples), 3)
        self.assertNotIn(target_id, samples)