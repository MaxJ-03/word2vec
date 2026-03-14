import numpy as np
from word2vec_base import Word2VecBase
from functions import Functions


# class to implement the skip-gram model
class SkipGram(Word2VecBase):
    
    def __init__(self, data_path, context_size, learning_rate, hidden_layer_size):
        super().__init__(data_path, context_size, learning_rate, hidden_layer_size)

        # Initialize weights with Xavier initialization
        limit_w1 = np.sqrt(6 / (self.V + self.N))
        self.W1 = np.random.uniform(-limit_w1, limit_w1, size=(self.V, self.N))
        limit_w2 = np.sqrt(6 / (self.V + self.N))
        self.W2 = np.random.uniform(-limit_w2, limit_w2, size=(self.N, self.V))
    

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