import numpy as np
import time
from word2vec_base import Word2VecBase
from functions import Functions

class CBOW(Word2VecBase):
    
    def __init__(self, data_path, context_size, learning_rate, hidden_layer_size):
        # Initializes the CBOW model parameters and applies Xavier initialization to the weight matrices.
        super().__init__(data_path, context_size, learning_rate, hidden_layer_size)

        # Initialize input embedding matrix uniformly, scaled by the embedding dimension to prevent vanishing weights.
        self.W1 = (np.random.rand(self.V, self.N) - 0.5) / self.N
        
        # Initialize output context matrix to zeros for stable initial predictions.
        self.W2 = np.zeros((self.N, self.V))

    def forward_pass(self, context_word_ids, context_size):
        # Performs the forward propagation to generate the target word probability distribution.
        
        # Aggregate the input context vectors by calculating their average to form the hidden layer.
        self.h = np.sum(self.W1[context_word_ids], axis=0) / context_size # h = (1/C) * W^T * (x_1 + ... + x_C) (Eq. 18)
        
        # Project the hidden layer state to the output vocabulary space.
        self.u = self.W2.T @ self.h # u_j = v'_w_j^T * h (Eq. 19)
        
        # Apply softmax to normalize the raw scores into a valid probability distribution.
        self.y_pred = Functions.softmax(self.u) # y_c = exp(u_c) / sum(exp(u_j)) (Eq. 20)
        
        return self.y_pred
    
    def backpropagation(self, output_word_id, context_word_ids, context_size):
        # Computes the prediction error and updates the network weights via gradient descent.
        
        # Determine the difference between the predicted distribution and the actual target word.
        e = self.y_pred.copy() # e_j = y_j - t_j (Eq. 21)
        e[output_word_id] -= 1

        # Backpropagate the error from the output layer to the hidden layer.
        EH = np.dot(self.W2, e) # dE/dh = sum(e_j * v'_w_j) (Eq. 24)

        # Update the hidden-to-output weights based on the hidden state and prediction error.
        self.W2 -= self.learning_rate * np.outer(self.h, e) # v'_w_j(new) = v'_w_j(old) - learning_rate * e_j * h (Eq. 23)

        # Update the input-to-hidden weights specifically for the context words used in the current sample.
        np.add.at(self.W1, context_word_ids, -(self.learning_rate / context_size) * EH) # v_w_I,c(new) = v_w_I,c(old) - (learning_rate / C) * dE/dh (Eq. 25)

    def train(self, epochs, print_interval=100000):
        # Runs the training loop across the dataset for the specified number of epochs.
        for epoch in range(epochs):
            epoch_loss = 0
            processed_samples = 0
            epoch_start_time = time.time()

            # Decay the learning rate to ensure smooth convergence as training progresses.
            self.update_learning_rate(epoch, epochs)

            for i in range(self.data_processing.data_length):
                
                # Extract the surrounding context words and the central target word for the current position.
                context_vector_ids, context_size = self.data_processing.one_hot_encoding_context_ids(i, self.context_size)
                output_word_id = self.data_processing.data_id_to_word_id(i)

                # Skip positions that have no context (e.g., single-token datasets).
                if context_size == 0:
                    continue
                
                # Execute the forward and backward passes to learn the word representations.
                predicted_vector = self.forward_pass(context_vector_ids, context_size)
                self.backpropagation(output_word_id, context_vector_ids, context_size)

                # Accumulate the cross-entropy loss to track model performance.
                epoch_loss += -np.log(predicted_vector[output_word_id] + 1e-9)
                processed_samples += 1
                
                if ((i + 1) % print_interval == 0): 
                    print(f'Epoch: {epoch + 1}, Trained first {i + 1} words of the dataset.')

            # Calculate the average loss over the entire dataset for the current epoch.
            average_loss = epoch_loss / max(processed_samples, 1)
            epoch_end_time = time.time()
            
            print(f"Epoch {epoch + 1}/{epochs} | LR: {self.learning_rate:.6f} | Average Loss: {average_loss:.4f} | Time: {(epoch_end_time - epoch_start_time):.2f}s")