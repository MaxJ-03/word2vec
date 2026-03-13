import numpy as np
import random, math
from dataprocessing import DataProcessing
from functions import Functions


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
    
    def backpropagation(self, input_vector_id, context_vectors):
        #calculate the error

        EI = np.zeros(self.V)

        for target_vector in context_vectors:
            EI += self.y_pred - target_vector

        EH = np.dot(self.W2, EI)

        self.W2 -= self.learning_rate * np.outer(self.h, EI)
        self.W1[input_vector_id] -= self.learning_rate * EH

    def train(self, epochs):

        for epoch in range(epochs):
            total_loss = 0
            for i in range(self.data_processing.data_length):

                input_vector_id = self.data_processing.data_id_to_word_id(i)
                context_vectors = self.data_processing.one_hot_encoding_context_list(i, self.context_size)

                predicted_vector = self.forward_pass(input_vector_id)

                self.backpropagation(input_vector_id, context_vectors)

                for target_vector in context_vectors:
                    total_loss += -np.sum(target_vector * np.log(predicted_vector))

            average_loss = total_loss / self.data_processing.data_length
            print(f"Epoch {epoch + 1}/{epochs} | Average Loss: {average_loss:.4f}")