import time
import numpy as np
from word2vec_base import Word2VecBase
from functions import Functions

class CBOWNegativeSampling(Word2VecBase):
    
    def __init__(self, data_path, context_size, learning_rate, hidden_layer_size, negative_sampling_size=20):
        # Initializes the CBOW model with Negative Sampling parameters and applies Xavier initialization.
        super().__init__(data_path, context_size, learning_rate, hidden_layer_size)

        self.negative_sampling_size = negative_sampling_size

        # Initialize input embedding matrix uniformly, scaled by the embedding dimension.
        self.W1 = (np.random.rand(self.V, self.N) - 0.5) / self.N
        
        # Initialize output context matrix to zeros.
        self.W2 = np.zeros((self.V, self.N))

    def update_weights(self, output_word_id, context_vector_ids, context_size):
        # Executes the forward and backward passes simultaneously for the true target and negative samples.
        
        # Aggregate the input context vectors by calculating their average to form the hidden layer.
        h = np.sum(self.W1[context_vector_ids], axis=0) / context_size # (Eq. 18)
        
        # Initialize the error accumulator for the hidden layer.
        EH = np.zeros(self.N)
        loss = 0
        
        # Compute the probability and gradient for the true positive target word.
        v_true = self.W2[output_word_id]
        f_true = Functions.sigmoid(np.dot(v_true, h))

        # Accumulate the log loss for the true positive prediction.
        loss -= np.log(f_true + 1e-9) # (Eq. 55)

        # Calculate the gradient for the true target word.
        gradient_true = f_true - 1 # (Eq. 57)

        # Backpropagate the error to the hidden layer accumulator.
        EH += gradient_true * v_true # (Eq. 59)

        # Update the output vector weights for the true target word.
        self.W2[output_word_id] -= self.learning_rate * gradient_true * h # (Eq. 58)

        # Generate negative samples to serve as incorrect target classes.
        negative_ids = self.data_processing.generate_negative_samples(output_word_id, self.negative_sampling_size)

        # Compute the probabilities and gradients for all negative samples.
        v_neg = self.W2[negative_ids]
        f_neg = Functions.sigmoid(np.dot(v_neg, h))

        # Accumulate the log loss for the negative sample predictions.
        loss -= np.sum(np.log(1 - f_neg + 1e-9)) # (Eq. 55)

        # Calculate the gradient for the negative samples.
        gradient_neg = f_neg - 0 # (Eq. 57)

        # Backpropagate the error from the negative samples to the hidden layer accumulator.
        EH += np.dot(gradient_neg, v_neg) # (Eq. 59)

        # Update the output vector weights for all selected negative samples.
        np.add.at(self.W2, negative_ids, -self.learning_rate * np.outer(gradient_neg, h)) # (Eq. 58)
    
        # Update the input-to-hidden weights specifically for the context words used in the current sample.
        np.add.at(self.W1, context_vector_ids, -(self.learning_rate / context_size) * EH) # (Eq. 60)

        return loss

    def train(self, epochs, print_interval=100000):
        # Runs the training loop across the dataset for the specified number of epochs.
        for epoch in range(epochs):
            
            epoch_loss = 0
            processed_samples = 0
            epoch_start_time = time.time()
            
            # Decay the learning rate to ensure smooth convergence as training progresses.
            self.update_learning_rate(epoch, epochs)

            for i in range(self.data_processing.data_length):

                # Extract the central target word and the surrounding context words for the current position.
                output_word_id = self.data_processing.data_id_to_word_id(i)
                context_vector_ids, context_size = self.data_processing.one_hot_encoding_context_ids(i, self.context_size)

                # Skip positions that have no context (e.g., single-token datasets).
                if context_size == 0:
                    continue

                # Compute gradients and apply weight updates for the current sample.
                loss = self.update_weights(output_word_id, context_vector_ids, context_size)

                # Accumulate the objective function loss to track model performance.
                epoch_loss += loss
                processed_samples += 1

                if ((i + 1) % print_interval == 0): 
                    print(f'Epoch: {epoch + 1}, Trained first {(i + 1)} words of the dataset.')

            # Calculate the average loss over the entire dataset for the current epoch.
            average_loss = epoch_loss / max(processed_samples, 1)
            epoch_end_time = time.time()
            
            print(f"Epoch {epoch + 1}/{epochs} | LR: {self.learning_rate:.6f} | Average Loss: {average_loss:.4f} | Time: {(epoch_end_time - epoch_start_time):.2f}s")