import numpy as np
from word2vec_base import Word2VecBase
from functions import Functions
from tree_utils import HuffmanTree

# class to implement the hierarchical softmax CBOW model
class CBOWHierarchical(Word2VecBase):
    
    def __init__(self, data_path, context_size, learning_rate, hidden_layer_size):
        super().__init__(data_path, context_size, learning_rate, hidden_layer_size)

        limit_w1 = np.sqrt(6 / (self.V + self.N))
        self.W1 = np.random.uniform(-limit_w1, limit_w1, size=(self.V, self.N))
        limit_w2 = np.sqrt(6 / (self.V-1 + self.N))
        self.W2 = np.random.uniform(-limit_w2, limit_w2, size=(self.V-1, self.N))
    
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