import numpy as np
import time
from word2vec_base import Word2VecBase
from functions import Functions


class SkipGramNegativeSampling(Word2VecBase):
    
    def __init__(self, data_path, context_size, learning_rate, hidden_layer_size, negative_sampling_size=5):
        # Initializes the Skip-Gram model with Negative Sampling parameters and applies Xavier initialization.
        super().__init__(data_path, context_size, learning_rate, hidden_layer_size)

        self.negative_sampling_size = negative_sampling_size

        # Initialize input embedding matrix uniformly, scaled by the embedding dimension.
        self.W1 = (np.random.rand(self.V, self.N) - 0.5) / self.N
        
        # Initialize output context matrix to zeros.
        self.W2 = np.zeros((self.V, self.N))

    def compute_gradients(self, input_vector_id, context_vector_ids):
        # Calculates loss and analytical gradients for the true context words and generated negative samples.

        # Extract the hidden layer representation directly from the input word matrix.
        h = self.W1[input_vector_id] # (Eq. 27)
        
        # Initialize the error accumulator for the hidden layer.
        EH = np.zeros(self.N)
        loss = 0
                
        # Compute the probability for the true positive context words.
        v_true = self.W2[context_vector_ids]
        f_true = Functions.sigmoid(np.dot(v_true, h))

        # Accumulate the log loss for the true positive predictions.
        loss -= np.sum(np.log(f_true + 1e-9)) # (Eq. 55)

        # Calculate the gradient for the true context words.
        gradient_true = f_true - 1 # (Eq. 57)

        # Backpropagate the error to the hidden layer accumulator.
        EH += np.dot(gradient_true, v_true) # (Eq. 59)

        dW2_true = np.outer(gradient_true, h)

        # Generate negative samples to serve as incorrect context classes.
        negative_ids = self.data_processing.generate_negative_samples_list(context_vector_ids, self.negative_sampling_size)

        # Compute the probabilities for all negative samples.
        v_neg = self.W2[negative_ids]
        f_neg = Functions.sigmoid(np.dot(v_neg, h))

        # Accumulate the log loss for the negative sample predictions.
        loss -= np.sum(np.log(1 - f_neg + 1e-9)) # (Eq. 55)

        # Calculate the gradient for the negative samples.
        gradient_neg = f_neg - 0 # (Eq. 57)

        # Backpropagate the error from the negative samples to the hidden layer accumulator.
        EH += np.dot(gradient_neg, v_neg) # (Eq. 59)

        dW2_neg = np.outer(gradient_neg, h)

        dW1 = EH

        return loss, dW2_true, dW2_neg, dW1, negative_ids

    def update_weights(self, input_vector_id, context_vector_ids):
        # Executes the forward and backward passes for the true context words and generated negative samples.
        
        loss, dW2_true, dW2_neg, dW1, negative_ids = self.compute_gradients(input_vector_id, context_vector_ids)

        # Update the output vector weights for the true context words.
        np.add.at(self.W2, context_vector_ids, -self.learning_rate * dW2_true) # (Eq. 58)

        # Update the output vector weights for all selected negative samples.
        np.add.at(self.W2, negative_ids, -self.learning_rate * dW2_neg) # (Eq. 58)

        # Update the input-to-hidden weight for the central input word.
        self.W1[input_vector_id] -= self.learning_rate * dW1 # (Eq. 60)

        return loss

    def train(self, epochs, print_interval=100000):
        # Runs the training loop across the dataset for the specified number of epochs.
        for epoch in range(epochs):

            epoch_loss = 0
            epoch_start_time = time.time()

            # Decay the learning rate to ensure smooth convergence as training progresses.
            self.update_learning_rate(epoch, epochs)

            for i in range(self.data_processing.data_length):

                # Extract the central input word and the surrounding target context words.
                input_vector_id = self.data_processing.data_id_to_word_id(i)
                context_vectors, context_size = self.data_processing.one_hot_encoding_context_ids(i, self.context_size)

                # Skip if there are no context words.
                if context_size == 0:
                    continue

                # Compute gradients and apply weight updates for the current sample.
                loss = self.update_weights(input_vector_id, context_vectors)

                # Accumulate the objective function loss to track model performance.
                epoch_loss += loss

                if ((i + 1) % print_interval == 0): 
                    print(f'Epoch: {epoch + 1}, Trained first {(i + 1)} words of the dataset.')

            # Calculate the average loss over the entire dataset for the current epoch.
            average_loss = epoch_loss / self.data_processing.data_length
            epoch_end_time = time.time()
            
            print(f"Epoch {epoch + 1}/{epochs} | LR: {self.learning_rate:.6f} | Average Loss: {average_loss:.4f} | Time: {(epoch_end_time - epoch_start_time):.2f}s")