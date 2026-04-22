import time
import numpy as np
from word2vec_base import Word2VecBase
from functions import Functions
from tree_utils import HuffmanTree

class SkipGramHierarchical(Word2VecBase):
    
    def __init__(self, data_path, context_size, learning_rate, hidden_layer_size):
        # Initializes the Skip-Gram model with Hierarchical Softmax and applies Xavier initialization.
        # W2 requires V-1 rows because a Huffman tree for V words has exactly V-1 internal nodes.
        super().__init__(data_path, context_size, learning_rate, hidden_layer_size)

        # Initialize input embedding matrix uniformly, scaled by the embedding dimension.
        self.W1 = (np.random.rand(self.V, self.N) - 0.5) / self.N
        
        # Initialize internal node weights for the Huffman tree to zeros.
        self.W2 = np.zeros((self.V - 1, self.N))

    def compute_gradients(self, input_vector_id, context_words, huffman_dict):
        # Calculates loss and analytical gradients by traversing the Huffman tree for each context word.

        # Extract the hidden layer representation directly from the input word matrix.
        h = self.W1[input_vector_id] # (Eq. 27)
        
        # Initialize the error accumulator for the hidden layer.
        EH = np.zeros(self.N)
        loss = 0

            
        # Extract the paths and codes for all context words.
        paths = [huffman_dict[w]['path'] for w in context_words]
        codes = [huffman_dict[w]['code'] for w in context_words]

        # Flatten the paths and codes to process all nodes across all context words in one batch.
        flat_paths = np.concatenate(paths)
        flat_codes = np.concatenate(codes)

        # Retrieve the weight vectors for the internal nodes along the path.
        v = self.W2[flat_paths]
            
        # Calculate the activation probabilities along the Huffman tree path.
        f = Functions.sigmoid(flat_codes * np.dot(v, h))

        # Accumulate the log loss for the path traversal.
        loss -= np.sum(np.log(f + 1e-9)) # (Eq. 51)
            
        # Calculate the gradient for the internal nodes along the path.
        gradient = flat_codes * (f - 1) # (Eq. 40)

        # Backpropagate the error from the internal nodes to the hidden layer accumulator.
        EH += np.dot(gradient, v) # (Eq. 42)

        dW2 = np.outer(gradient, h)

        dW1 = EH 

        return loss, dW2, dW1, flat_paths
    
    def update_weights(self, input_vector_id, context_words, huffman_dict):
        # Executes the forward and backward passes by traversing the Huffman tree for each context word.
        
        loss, dW2, dW1, flat_paths = self.compute_gradients(input_vector_id, context_words, huffman_dict)
        
        # Accumulate the updates for all internal nodes.
        np.add.at(self.W2, flat_paths, -self.learning_rate * dW2) # (Eq. 43)
            
        # Update the input-to-hidden weight for the central input word.
        self.W1[input_vector_id] -= self.learning_rate * dW1 # (Eq. 52)

        return loss

    def train(self, epochs, print_interval=1000000):
        # Builds the Huffman tree and runs the training loop across the dataset.
        
        # Construct the binary Huffman tree based on vocabulary frequencies.
        huffman_tree = HuffmanTree(self.data_processing.vocabulary, self.data_processing.vocabulary_frequency)
        huffman_dict = huffman_tree.convert_tree_to_dict(huffman_tree.root)

        print("Tree created.")

        for epoch in range(epochs):

            epoch_loss = 0
            epoch_start_time = time.time()

            # Decay the learning rate to ensure smooth convergence as training progresses.
            self.update_learning_rate(epoch, epochs)

            for i in range(self.data_processing.data_length):

                # Extract the central input word and the surrounding target context words.
                input_vector_id = self.data_processing.data_id_to_word_id(i)
                context_words, context_size = self.data_processing.one_hot_encoding_context_words(i, self.context_size)

                # Skip if there are no context words.
                if context_size == 0:
                    continue

                # Compute gradients and apply weight updates by traversing the Huffman tree for each target.
                loss = self.update_weights(input_vector_id, context_words, huffman_dict)

                # Accumulate the objective function loss to track model performance.
                epoch_loss += loss

                if ((i + 1) % print_interval == 0):
                    print(f'Epoch: {epoch + 1}, Trained first {(i + 1)} words of the dataset.')

            # Calculate the average loss over the entire dataset for the current epoch.
            average_loss = epoch_loss / self.data_processing.data_length
            epoch_end_time = time.time()
            
            print(f"Epoch {epoch + 1}/{epochs} | LR: {self.learning_rate:.6f} | Average Loss: {average_loss:.4f} | Time: {(epoch_end_time - epoch_start_time):.2f}s")