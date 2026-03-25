#class including all the preprocessing steps for the data
import re
import numpy as np
import math
from collections import Counter

class DataProcessing:
    def __init__(self):
        self.data = None
        self.data_length = None
        self.tokenized_data = None
        self.vocabulary = None
        self.vocabulary_frequency = None
        self.word_to_id = None
        self.id_to_word = None
        self.probability_distribution = None
        self.unigram_table = None

    def load(self, path):
        #load the data from txt file at the path
        with open(path, 'r', encoding='utf-8') as file:
            self.data = file.read()
        self.tokenize()
        self.create_vocabulary()
        self.word_id_mapping()
        self.create_negative_sampling_distribution()

    def tokenize(self):
        #tokenize the data by splitting it into words
        pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
        self.tokenized_data = pattern.findall(self.data.lower())
        self.data_length = len(self.tokenized_data)
        print(f"Dataset Size: {self.data_length}")
    
    def create_vocabulary(self):
        #tokenize the data by splitting it into words
        self.vocabulary = sorted(list(set(self.tokenized_data)))
        #create a frequency dictionary for the vocabulary
        self.vocabulary_frequency = Counter(self.tokenized_data)
        print(f"Vocabulary Size: {len(self.vocabulary)}")

    def word_id_mapping(self):
        #create a mapping of words to ids
        self.word_to_id = {word: i for i, word in enumerate(self.vocabulary)}
        self.id_to_word = {i: word for i, word in enumerate(self.vocabulary)}
    
    def one_hot_encoding(self, word):
        #one hot encode the word
        one_hot_vector = np.zeros(len(self.vocabulary))
        one_hot_vector[self.word_to_id[word]] = 1
        return one_hot_vector
    
    def one_hot_encoding_id(self, id):
        #one hot encode the word by id
        one_hot_vector = np.zeros(len(self.vocabulary))
        one_hot_vector[id] = 1
        return one_hot_vector
    
    def one_hot_encoding_context(self, id, context_size):
        #one hot encode the words in the window
        start = max(0, id - math.floor(context_size / 2))
        end = min(self.data_length, id + math.ceil(context_size / 2) + 1)
        window_words = self.tokenized_data[start:id] + self.tokenized_data[id+1:end]
        #merge all vectors in 1 vector by summing them up
        context_vectors = [self.one_hot_encoding(word) for word in window_words]
        actual_context_size = len(context_vectors)
        return np.sum(context_vectors, axis=0), actual_context_size
    
    def one_hot_encoding_context_list(self, id, context_size):
        #one hot encode the words in the window
        start = max(0, id - math.floor(context_size / 2))
        end = min(self.data_length, id + math.ceil(context_size / 2) + 1)
        window_words = self.tokenized_data[start:id] + self.tokenized_data[id+1:end]
        context_vectors = [self.one_hot_encoding(word) for word in window_words]
        actual_context_size = len(context_vectors)
        return context_vectors
    
    def one_hot_encoding_context_ids(self, id, context_size):
        #one hot encode the words in the window
        start = max(0, id - math.floor(context_size / 2))
        end = min(self.data_length, id + math.ceil(context_size / 2) + 1)
        window_words = self.tokenized_data[start:id] + self.tokenized_data[id+1:end]
        context_vector_ids = [self.word_to_id[word] for word in window_words]
        actual_context_size = len(context_vector_ids)
        return context_vector_ids, actual_context_size
    
    def one_hot_encoding_context_words(self, id, context_size):
        #one hot encode the words in the window
        start = max(0, id - math.floor(context_size / 2))
        end = min(self.data_length, id + math.ceil(context_size / 2) + 1)
        window_words = self.tokenized_data[start:id] + self.tokenized_data[id+1:end]
        actual_context_size = len(window_words)
        return window_words, actual_context_size
    
    def one_hot_encoding_to_word(self, one_hot_vector):
        #convert one hot vector back to word
        id = np.argmax(one_hot_vector)
        return self.id_to_word[id]
    
    def one_hot_encoding_to_words(self, one_hot_vector):
        #convert one hot vector back to words in the window
        ids = np.where(one_hot_vector == 1)[0]
        return [self.id_to_word[id] for id in ids]

    def one_hot_encoding_to_id(self, one_hot_vector):
        #convert one hot vector back to id
        id = np.argmax(one_hot_vector)
        return id

    def data_id_to_word_id(self, id):
        #convert data id to word id
        word = self.tokenized_data[id]
        return self.word_to_id[word]

    def create_negative_sampling_distribution(self):
        # Build frequencies in vocabulary-id order so sampled ids map to the correct words.
        modified_frequencies = np.array(
            [self.vocabulary_frequency[word] for word in self.vocabulary], dtype=float
        ) ** 0.75
        total_sum = np.sum(modified_frequencies)
        self.probability_distribution = modified_frequencies / total_sum

        table_size = 1e6
        unigram_table = []

        for word, prob in enumerate(self.probability_distribution):
            count = int(prob * table_size)
            unigram_table.extend([word] * count)

        self.unigram_table = np.array(unigram_table)

    def generate_negative_samples(self, target_id, num_samples):
        #generate negative samples based on the probability distribution
        negative_samples = [target_id]
        while target_id in negative_samples:
            negative_samples = np.random.choice(self.unigram_table, size=num_samples, replace=True).tolist()
        
        return negative_samples
    
    def generate_negative_samples_list(self, target_ids, num_samples):

        total_samples = num_samples * len(target_ids)

        #generate negative samples based on the probability distribution
        choice_indices = np.random.randint(0, len(self.unigram_table), size=total_samples)
        negative_samples = self.unigram_table[choice_indices].tolist()

        for i in range(len(negative_samples)):
            while negative_samples[i] in target_ids:
                new_index = np.random.randint(0, len(self.unigram_table))
                negative_samples[i] = self.unigram_table[new_index]

        return negative_samples

