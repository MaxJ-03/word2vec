import numpy as np
import time
from word2vec_base import Word2VecBase
from functions import Functions
from tree_utils import HuffmanTree

class CBOWHierarchical(Word2VecBase):
    
    def __init__(self, data_path, context_size, learning_rate, hidden_layer_size):
        # Initializes the CBOW model with Hierarchical Softmax and applies Xavier initialization.
        # W2 requires V-1 rows because a Huffman tree for V words has exactly V-1 internal nodes.
        super().__init__(data_path, context_size, learning_rate, hidden_layer_size)

        # Calculate the boundary for Xavier/Glorot initialization to maintain variance across layers.
        limit_w1 = np.sqrt(6 / (self.V + self.N))
        self.W1 = np.random.uniform(-limit_w1, limit_w1, size=(self.V, self.N))
        
        limit_w2 = np.sqrt(6 / (self.V - 1 + self.N))
        self.W2 = np.random.uniform(-limit_w2, limit_w2, size=(self.V - 1, self.N))
    
    def update_weights(self, output_word_id, context_vector_ids, context_size, huffman_dict):
        # Executes the forward and backward passes traversing the Huffman tree.
        
        # Aggregate the input context vectors by calculating their average to form the hidden layer.
        h = np.sum(self.W1[context_vector_ids], axis=0) / context_size # (Eq. 18)

        # Extract the Huffman tree path and corresponding routing codes for the target word.
        node = huffman_dict[self.data_processing.id_to_word[output_word_id]]
        code = np.array(node['code'])
        path = node['path']

        # Retrieve the weight vectors for the internal nodes along the path.
        v = self.W2[path]
        
        # Calculate the activation probabilities along the Huffman tree path.
        f = Functions.sigmoid(code * np.dot(v, h))

        # Accumulate the log loss for the path traversal.
        loss = -np.sum(np.log(f + 1e-9)) # (Eq. 39)

        # Calculate the gradient for the internal nodes along the path.
        gradient = code * (f - 1) # (Eq. 40)

        # Backpropagate the error from the internal nodes to the hidden layer accumulator.
        EH = np.dot(gradient, v) # (Eq. 42)

        # Update the weight vectors of the internal nodes along the target path.
        self.W2[path] -= self.learning_rate * np.outer(gradient, h) # (Eq. 43)
    
        # Update the input-to-hidden weights specifically for the context words used.
        np.add.at(self.W1, context_vector_ids, -(self.learning_rate / context_size) * EH) # (Eq. 44)

        return loss

    def train(self, epochs, print_interval=100000):
        # Builds the Huffman tree and runs the training loop across the dataset.
        
        # Construct the binary Huffman tree based on vocabulary frequencies.
        huffman_tree = HuffmanTree(self.data_processing.vocabulary, self.data_processing.vocabulary_frequency)
        huffman_dict = huffman_tree.convert_tree_to_dict(huffman_tree.root)

        print("Tree created.")

        for epoch in range(epochs):

            epoch_loss = 0
            processed_samples = 0
            epoch_start_time = time.time()
            
            # Decay the learning rate to ensure smooth convergence as training progresses.
            self.update_learning_rate(epoch, epochs)

            for i in range(self.data_processing.data_length):

                # Extract the central target word and the surrounding context words.
                output_word_id = self.data_processing.data_id_to_word_id(i)
                context_vector_ids, context_size = self.data_processing.one_hot_encoding_context_ids(i, self.context_size)

                # Skip positions that have no context (e.g., single-token datasets).
                if context_size == 0:
                    continue

                # Compute gradients and apply weight updates by traversing the Huffman tree.
                loss = self.update_weights(output_word_id, context_vector_ids, context_size, huffman_dict)

                # Accumulate the objective function loss to track model performance.
                epoch_loss += loss
                processed_samples += 1

                if ((i + 1) % print_interval == 0): 
                    print(f'Epoch: {epoch + 1}, Trained first {i + 1} words of the dataset.')

            # Calculate the average loss over the entire dataset for the current epoch.
            average_loss = epoch_loss / max(processed_samples, 1)
            epoch_end_time = time.time()
            
            print(f"Epoch {epoch + 1}/{epochs} | LR: {self.learning_rate:.6f} | Average Loss: {average_loss:.4f} | Time: {(epoch_end_time - epoch_start_time):.2f}s")