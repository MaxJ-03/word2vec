import numpy as np
import random
from dataprocessing import DataProcessing
from functions import Functions

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
    

    def forward_pass(self, context_vector, context_size):

        average_context_vector = context_vector
        #forward pass through the network
        self.h = self.W1.T @ average_context_vector
        self.h /= context_size
        self.u = self.W2.T @ self.h
        self.y_pred = Functions.softmax(self.u)
        return self.y_pred
    
    def backpropagation(self, output_vector, context_vector, context_size):
        #calculate the error
        e = self.y_pred - output_vector

        EH = np.dot(self.W2, e)

        self.W2 -= self.learning_rate * np.outer(self.h, e)
        self.W1 -= (1 / context_size) * self.learning_rate * np.outer(context_vector, EH)

    def train(self, epochs):

        for epoch in range(epochs):
            epoch_loss = 0
            counter = 1
            for i in range(self.data_processing.data_length):

                context_vector, actual_context_size = self.data_processing.one_hot_encoding_context(i, self.context_size)
                
                output_vector = self.data_processing.one_hot_encoding(self.data_processing.tokenized_data[i])

                predicted_vector = self.forward_pass(context_vector, actual_context_size)

                self.backpropagation(output_vector, context_vector, actual_context_size)

                epoch_loss += -np.sum(output_vector * np.log(predicted_vector))
                
                if (counter % 1000 == 0): print(f'Epoch: {epoch + 1}, Trained first {counter} words of the dataset.')
                counter+=1

            average_loss = epoch_loss / self.data_processing.data_length
            print(f"Epoch {epoch + 1}/{epochs} | Average Loss: {average_loss:.4f}")