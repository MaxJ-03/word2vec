import numpy as np
import random
from dataprocessing import DataProcessing
from functions import Functions
from tree_utils import HuffmanTree

# class to implement the CBOW model
class CBOW:
    
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
    

    def forward_pass(self, context_word_ids, context_size):

        self.h = np.sum(self.W1[context_word_ids], axis=0) / context_size
        #forward pass through the network
        self.u = self.W2.T @ self.h
        self.y_pred = Functions.softmax(self.u)
        return self.y_pred
    
    def backpropagation(self, output_word_id, context_word_ids, context_size):
        #calculate the error
        e = self.y_pred.copy()
        e[output_word_id] -= 1

        EH = np.dot(self.W2, e)

        self.W2 -= self.learning_rate * np.outer(self.h, e)

        np.add.at(self.W1, context_word_ids, -(self.learning_rate / context_size) * EH)

    def train(self, epochs):

        for epoch in range(epochs):
            epoch_loss = 0
            counter = 1
            for i in range(self.data_processing.data_length):

                context_vector_ids, context_size = self.data_processing.one_hot_encoding_context_ids(i, self.context_size)
                
                output_word_id = self.data_processing.data_id_to_word_id(i)
                
                predicted_vector = self.forward_pass(context_vector_ids, context_size)

                self.backpropagation(output_word_id, context_vector_ids, context_size)

                epoch_loss += -np.log(predicted_vector[output_word_id] + 1e-9)
                
                if (counter % 1000 == 0): print(f'Epoch: {epoch + 1}, Trained first {counter} words of the dataset.')
                counter+=1

            average_loss = epoch_loss / self.data_processing.data_length
            print(f"Epoch {epoch + 1}/{epochs} | Average Loss: {average_loss:.4f}")

# class to implement the CBOW model
class CBOWHierarchical:
    
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
    
    def update_weights(self, output_word_id, context_vector_ids, context_size, huffman_dict):

        h = np.sum(self.W1[context_vector_ids], axis=0) / context_size
        
        EH = np.zeros(self.N)

        window_loss = 0

        node = huffman_dict[self.data_processing.id_to_word[output_word_id]]
        code = np.array(node['code'])
        path = node['path']

        v = self.W2[path]
        f = Functions.sigmoid(code * np.dot(v, h))

        loss = -np.sum(np.log(f + 1e-9))

        gradient = code * (f - 1)

        EH = np.dot(gradient, v)

        self.W2[path] -= self.learning_rate * np.outer(gradient, h)
    
        np.add.at(self.W1, context_vector_ids, -(self.learning_rate / context_size) * EH)

        return loss

    def train(self, epochs):

        huffman_tree = HuffmanTree(self.data_processing.vocabulary, self.data_processing.vocabulary_frequency)
        huffman_dict = huffman_tree.convert_tree_to_dict(huffman_tree.root)

        print("Tree created!")

        for epoch in range(epochs):

            epoch_loss = 0
            counter = 1 

            for i in range(self.data_processing.data_length):

                output_word_id = self.data_processing.data_id_to_word_id(i)
                context_vector_ids, context_size = self.data_processing.one_hot_encoding_context_ids(i, self.context_size)

                loss = self.update_weights(output_word_id, context_vector_ids, context_size, huffman_dict)

                epoch_loss += loss

                if (counter % 1000 == 0): print(f'Epoch: {epoch + 1}, Trained first {counter} words of the dataset.')
                counter+=1

            average_loss = epoch_loss / self.data_processing.data_length
            print(f"Epoch {epoch + 1}/{epochs} | Average Loss: {average_loss:.4f}")


# class to implement the CBOW model
class CBOWNegativeSampling:
    
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
    
    def update_weights(self, output_word_id, context_vector_ids, context_size):

        h = np.sum(self.W1[context_vector_ids], axis=0) / context_size
        
        EH = np.zeros(self.N)

        loss = 0
        
        v_true = self.W2[output_word_id]
        f_true = Functions.sigmoid(np.dot(v_true, h))

        loss -= np.log(f_true + 1e-9)

        gradient_true = f_true - 1

        EH += gradient_true * v_true

        self.W2[output_word_id] -= self.learning_rate * gradient_true * h

        negative_ids = self.data_processing.generate_negative_samples(output_word_id, self.negative_sampling_size)

        v_neg = self.W2[negative_ids]

        f_neg = Functions.sigmoid(np.dot(v_neg, h))

        loss -= np.sum(np.log(1 - f_neg + 1e-9))

        gradient_neg = f_neg - 0

        EH += np.dot(gradient_neg, v_neg)

        np.add.at(self.W2, negative_ids, -self.learning_rate * np.outer(gradient_neg, h))       
    
        np.add.at(self.W1, context_vector_ids, -(self.learning_rate / context_size) * EH)

        return loss

    def train(self, epochs):

        for epoch in range(epochs):

            epoch_loss = 0
            counter = 1 

            for i in range(self.data_processing.data_length):

                output_word_id = self.data_processing.data_id_to_word_id(i)
                context_vector_ids, context_size = self.data_processing.one_hot_encoding_context_ids(i, self.context_size)

                loss = self.update_weights(output_word_id, context_vector_ids, context_size)

                epoch_loss += loss

                if (counter % 1000 == 0): print(f'Epoch: {epoch + 1}, Trained first {counter} words of the dataset.')
                counter+=1

            average_loss = epoch_loss / self.data_processing.data_length
            print(f"Epoch {epoch + 1}/{epochs} | Average Loss: {average_loss:.4f}")