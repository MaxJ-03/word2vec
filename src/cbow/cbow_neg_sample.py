import time

import numpy as np
from word2vec_base import Word2VecBase
from functions import Functions

# class to implement the negative CBOW model
class CBOWNegativeSampling(Word2VecBase):
    
    def __init__(self, data_path, context_size, learning_rate, hidden_layer_size, negative_sampling_size = 20):
        super().__init__(data_path, context_size, learning_rate, hidden_layer_size)

        self.negative_sampling_size = negative_sampling_size

        # Initialize weights with Xavier initialization
        limit_w1 = np.sqrt(6 / (self.V + self.N))
        self.W1 = np.random.uniform(-limit_w1, limit_w1, size=(self.V, self.N))
        limit_w2 = np.sqrt(6 / (self.V + self.N))
        self.W2 = np.random.uniform(-limit_w2, limit_w2, size=(self.V, self.N))

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
            epoch_start_time = time.time()
            self.update_learning_rate(epoch, epochs)

            for i in range(self.data_processing.data_length):

                output_word_id = self.data_processing.data_id_to_word_id(i)
                context_vector_ids, context_size = self.data_processing.one_hot_encoding_context_ids(i, self.context_size)

                loss = self.update_weights(output_word_id, context_vector_ids, context_size)

                epoch_loss += loss

                if ((i + 1) % 100000 == 0): print(f'Epoch: {epoch + 1}, Trained first {(i + 1)} words of the dataset.')

            average_loss = epoch_loss / self.data_processing.data_length
            epoch_end_time = time.time()
            print(f"Epoch {epoch + 1}/{epochs} | LR: {self.learning_rate:.6f} | Average Loss: {average_loss:.4f} | Time: {(epoch_end_time - epoch_start_time):.2f}s")