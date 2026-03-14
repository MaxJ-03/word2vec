import numpy as np
from word2vec_base import Word2VecBase
from functions import Functions


# class to implement the skip-gram negative sampling model
class SkipGramNegativeSampling(Word2VecBase):
    
    def __init__(self, data_path, context_size, learning_rate, hidden_layer_size, negative_sampling_size = 5):
        super().__init__(data_path, context_size, learning_rate, hidden_layer_size)

        self.negative_sampling_size = negative_sampling_size

        # Initialize weights with Xavier initialization
        limit_w1 = np.sqrt(6 / (self.V + self.N))
        self.W1 = np.random.uniform(-limit_w1, limit_w1, size=(self.V, self.N))
        limit_w2 = np.sqrt(6 / (self.V + self.N))
        self.W2 = np.random.uniform(-limit_w2, limit_w2, size=(self.V, self.N))
    
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