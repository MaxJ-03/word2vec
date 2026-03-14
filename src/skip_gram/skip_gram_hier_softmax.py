import numpy as np
from word2vec_base import Word2VecBase
from functions import Functions
from tree_utils import HuffmanTree

# class to implement the skip-gram hierarchical softmax model
class SkipGramHierarchical(Word2VecBase):
    
    def __init__(self, data_path, context_size, learning_rate, hidden_layer_size):
        super().__init__(data_path, context_size, learning_rate, hidden_layer_size)

        # Initialize weights with Xavier initialization
        limit_w1 = np.sqrt(6 / (self.V + self.N))
        self.W1 = np.random.uniform(-limit_w1, limit_w1, size=(self.V, self.N))
        limit_w2 = np.sqrt(6 / (self.V + self.N))
        self.W2 = np.random.uniform(-limit_w2, limit_w2, size=(self.V-1, self.N))
    
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


