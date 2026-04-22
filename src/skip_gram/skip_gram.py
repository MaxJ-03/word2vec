import time
import numpy as np
from word2vec_base import Word2VecBase
from functions import Functions


class SkipGram(Word2VecBase):
    
    def __init__(self, data_path, context_size, learning_rate, hidden_layer_size):
        # Initializes the Skip-Gram model parameters and applies Xavier initialization to the weight matrices.
        super().__init__(data_path, context_size, learning_rate, hidden_layer_size)

        # Initialize input embedding matrix uniformly, scaled by the embedding dimension.
        self.W1 = (np.random.rand(self.V, self.N) - 0.5) / self.N
        
        # Initialize output context matrix to zeros.
        self.W2 = np.zeros((self.N, self.V))
    

    def forward_pass(self, input_vector_id):
        # Performs the forward propagation to generate the context word probability distribution.
        
        # Extract the hidden layer representation directly from the input word matrix.
        self.h = self.W1[input_vector_id] # h = v_w_I (Eq. 27)
        
        # Project the hidden layer state to the output vocabulary space.
        self.u = self.W2.T @ self.h # u_j = v'_w_j^T * h (Eq. 29)
        
        # Apply softmax to normalize the raw scores into a valid probability distribution.
        self.y_pred = Functions.softmax(self.u) # y_{c,j} = exp(u_{c,j}) / sum(exp(u_j')) (Eq. 30)
        
        return self.y_pred
    
    def compute_gradients(self, input_vector_id, context_vectors_ids, context_size):
        # Calculates the forward pass, loss, and analytical gradients without updating weights.

        y_pred = self.forward_pass(input_vector_id)
        loss = -np.sum(np.log(y_pred[context_vectors_ids] + 1e-9))

        # Calculate the sum of prediction errors for all words in the context window.
        EI = context_size * y_pred # EI_j = C * y_j - sum(t_{c,j}) (Eq. 34)
        np.add.at(EI, context_vectors_ids, -1) 

        # Backpropagate the accumulated error from the output layer to the hidden layer.
        EH = np.dot(self.W2, EI) # dE/dh = sum(EI_j * v'_w_j) (Eq. 36)

        dW2 = np.outer(self.h, EI) # dE/dv'_w_j = EI_j * h (Eq. 35)
        dW1 = EH # dE/dv_w_I = dE/dh (Eq. 38)

        return loss, dW2, dW1

    def backpropagation(self, input_vector_id, context_vectors_ids, context_size):
        # Computes the accumulated prediction error across all context words and updates the network weights.

        loss, dW2, dW1 = self.compute_gradients(input_vector_id, context_vectors_ids, context_size)

        # Update the hidden-to-output weights based on the hidden state and accumulated error.
        self.W2 -= self.learning_rate * dW2 # v'_w_j(new) = v'_w_j(old) - learning_rate * EI_j * h (Eq. 37)

        # Update the input-to-hidden weight specifically for the central input word used in the current sample.
        self.W1[input_vector_id] -= self.learning_rate * dW1 # v_w_I(new) = v_w_I(old) - learning_rate * dE/dh (Eq. 38)

        return loss

    def train(self, epochs, print_interval=100000):
        # Runs the training loop across the dataset for the specified number of epochs.
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_start_time = time.time()

            # Decay the learning rate to ensure smooth convergence as training progresses.
            self.update_learning_rate(epoch, epochs)

            for i in range(self.data_processing.data_length):

                # Extract the central input word and the surrounding target context words for the current position.
                input_vector_id = self.data_processing.data_id_to_word_id(i)
                context_vectors_ids, context_size = self.data_processing.one_hot_encoding_context_ids(i, self.context_size)

                # Skip if there are no context words.
                if context_size == 0:
                    continue

                # Execute the forward and backward passes to learn the word representations.
                loss = self.backpropagation(input_vector_id, context_vectors_ids, context_size)

                # Accumulate the sum of log probabilities for the actual context words to track model performance.
                epoch_loss += loss # (Eq. 26)

                if ((i + 1) % print_interval == 0): 
                    print(f'Epoch: {epoch + 1}, Trained first {(i + 1)} words of the dataset.')

            # Calculate the average loss over the entire dataset for the current epoch.
            average_loss = epoch_loss / self.data_processing.data_length
            epoch_end_time = time.time()
            
            print(f"Epoch {epoch + 1}/{epochs} | LR: {self.learning_rate:.6f} | Average Loss: {average_loss:.4f} | Time: {(epoch_end_time - epoch_start_time):.2f}s")