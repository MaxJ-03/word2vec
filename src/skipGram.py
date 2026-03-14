import numpy as np
import random, math
from dataprocessing import DataProcessing
from functions import Functions
from tree_utils import HuffmanTree


# class to implement the skip-gram model
class SkipGram:
    
    def __init__(self, data_path, context_size, learning_rate, hidden_layer_size):
        self.context_size = context_size
        self.learning_rate = learning_rate
        self.data_processing = DataProcessing()
        self.data_processing.load(data_path)
        #size of the vocabulary
        self.V = len(self.data_processing.vocabulary)
        #size of hidden layer
        self.N = hidden_layer_size
        self.W1 = np.random.uniform(-0.1, 0.1, size=(self.V, self.N))
        self.W2 = np.random.uniform(-0.1, 0.1, size=(self.N, self.V))
    

    def forward_pass(self, input_vector_id):

        #forward pass through the network
        self.h = self.W1[input_vector_id]
        self.u = self.W2.T @ self.h
        self.y_pred = Functions.softmax(self.u)
        return self.y_pred
    
    def backpropagation(self, input_vector_id, context_vectors_ids, context_size):
        #calculate the error

        EI = context_size * self.y_pred

        np.add.at(EI, context_vectors_ids, -1)

        EH = np.dot(self.W2, EI)

        self.W2 -= self.learning_rate * np.outer(self.h, EI)
        self.W1[input_vector_id] -= self.learning_rate * EH

    def train(self, epochs):

        for epoch in range(epochs):
            total_loss = 0
            for i in range(self.data_processing.data_length):

                input_vector_id = self.data_processing.data_id_to_word_id(i)
                context_vectors_ids, context_size = self.data_processing.one_hot_encoding_context_ids(i, self.context_size)

                predicted_vector = self.forward_pass(input_vector_id)

                self.backpropagation(input_vector_id, context_vectors_ids, context_size)

                total_loss += -np.sum(np.log(predicted_vector[context_vectors_ids] + 1e-9))

            average_loss = total_loss / self.data_processing.data_length
            print(f"Epoch {epoch + 1}/{epochs} | Average Loss: {average_loss:.4f}")


# class to implement the skip-gram model
class SkipGramHierarchical:
    
    def __init__(self, data_path, context_size, learning_rate, hidden_layer_size):
        self.context_size = context_size
        self.learning_rate = learning_rate
        self.data_processing = DataProcessing()
        self.data_processing.load(data_path)
        #size of the vocabulary
        self.V = len(self.data_processing.vocabulary)
        #size of hidden layer
        self.N = hidden_layer_size
        self.W1 = np.random.uniform(-0.1, 0.1, size=(self.V, self.N))
        self.W2 = np.random.uniform(-0.1, 0.1, size=(self.V-1, self.N))
    
    def update_weights(self, input_vector_id, context_words, huffman_dict):

        h = self.W1[input_vector_id]
        
        EH = np.zeros(self.N)

        loss = 0

        for target_word in context_words:
            node = huffman_dict[target_word]
            code = np.array(node['code'])
            path = node['path']

            v = self.W2[path]
            f = Functions.sigmoid(code * np.dot(v, h))

            loss -= np.sum(np.log(f + 1e-9))
            
            gradient = code * (f - 1)

            EH += np.dot(gradient, v)

            self.W2[path] -= self.learning_rate * np.outer(gradient, h)
    
        self.W1[input_vector_id] -= self.learning_rate * EH

        return loss

    def train(self, epochs):

        huffman_tree = HuffmanTree(self.data_processing.vocabulary, self.data_processing.vocabulary_frequency)
        huffman_dict = huffman_tree.convert_tree_to_dict(huffman_tree.root)

        for epoch in range(epochs):

            epoch_loss = 0

            for i in range(self.data_processing.data_length):

                input_vector_id = self.data_processing.data_id_to_word_id(i)
                context_words, context_size = self.data_processing.one_hot_encoding_context_words(i, self.context_size)

                loss = self.update_weights(input_vector_id, context_words, huffman_dict)

                epoch_loss += loss

            average_loss = epoch_loss / self.data_processing.data_length
            print(f"Epoch {epoch + 1}/{epochs} | Average Loss: {average_loss:.4f}")


# class to implement the skip-gram model
class SkipGramNegativeSampling:
    
    def __init__(self, data_path, context_size, learning_rate, hidden_layer_size, negative_sampling_size = 5):
        self.context_size = context_size
        self.learning_rate = learning_rate
        self.data_processing = DataProcessing()
        self.data_processing.load(data_path)
        self.negative_sampling_size = negative_sampling_size
        #size of the vocabulary
        self.V = len(self.data_processing.vocabulary)
        #size of hidden layer
        self.N = hidden_layer_size
        self.W1 = np.random.uniform(-0.1, 0.1, size=(self.V, self.N))
        self.W2 = np.random.uniform(-0.1, 0.1, size=(self.V, self.N))
    
    def update_weights(self, input_vector_id, context_vector_ids):

        h = self.W1[input_vector_id]
        
        EH = np.zeros(self.N)

        loss = 0
                
        v_true = self.W2[context_vector_ids]
        f_true = Functions.sigmoid(np.dot(v_true, h))

        loss -= np.sum(np.log(f_true + 1e-9))

        gradient_true = f_true - 1

        EH += np.dot(gradient_true, v_true)

        np.add.at(self.W2, context_vector_ids, -self.learning_rate * np.outer(gradient_true, h))

        negative_ids = self.data_processing.generate_negative_samples_list(context_vector_ids, self.negative_sampling_size)

        v_neg = self.W2[negative_ids]
        f_neg = Functions.sigmoid(np.dot(v_neg, h))

        loss -= np.sum(np.log(1 - f_neg + 1e-9))

        gradient_neg = f_neg - 0

        EH += np.dot(gradient_neg, v_neg)

        np.add.at(self.W2, negative_ids, -self.learning_rate * np.outer(gradient_neg, h))

        self.W1[input_vector_id] -= self.learning_rate * EH

        return loss

    def train(self, epochs):

        for epoch in range(epochs):

            epoch_loss = 0

            for i in range(self.data_processing.data_length):

                input_vector_id = self.data_processing.data_id_to_word_id(i)
                context_vectors, context_size = self.data_processing.one_hot_encoding_context_ids(i, self.context_size)

                loss = self.update_weights(input_vector_id, context_vectors)

                epoch_loss += loss

            average_loss = epoch_loss / self.data_processing.data_length
            print(f"Epoch {epoch + 1}/{epochs} | Average Loss: {average_loss:.4f}")