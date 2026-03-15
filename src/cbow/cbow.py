import numpy as np
import time
from word2vec_base import Word2VecBase
from functions import Functions

# class to implement the CBOW model
class CBOW(Word2VecBase):
    
    def __init__(self, data_path, context_size, learning_rate, hidden_layer_size):
        super().__init__(data_path, context_size, learning_rate, hidden_layer_size)

        # Initialize weights with Xavier initialization
        limit_w1 = np.sqrt(6 / (self.V + self.N))
        self.W1 = np.random.uniform(-limit_w1, limit_w1, size=(self.V, self.N))
        limit_w2 = np.sqrt(6 / (self.N + self.V))
        self.W2 = np.random.uniform(-limit_w2, limit_w2, size=(self.N, self.V))

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

            epoch_start_time = time.time()

            self.update_learning_rate(epoch, epochs)

            for i in range(self.data_processing.data_length):

                context_vector_ids, context_size = self.data_processing.one_hot_encoding_context_ids(i, self.context_size)
                
                output_word_id = self.data_processing.data_id_to_word_id(i)
                
                predicted_vector = self.forward_pass(context_vector_ids, context_size)

                self.backpropagation(output_word_id, context_vector_ids, context_size)

                epoch_loss += -np.log(predicted_vector[output_word_id] + 1e-9)
                
                if ((i + 1) % 1000 == 0): print(f'Epoch: {epoch + 1}, Trained first {i + 1} words of the dataset.')

            average_loss = epoch_loss / self.data_processing.data_length
            epoch_end_time = time.time()
            print(f"Epoch {epoch + 1}/{epochs} | LR: {self.learning_rate:.6f} | Average Loss: {average_loss:.4f} | Time: {(epoch_end_time - epoch_start_time):.2f}s")