import numpy as np
import time
from word2vec_base import Word2VecBase
from functions import Functions


class SkipGramNegativeSampling(Word2VecBase):
    
    def __init__(self, data_path, context_size, learning_rate, hidden_layer_size, negative_sampling_size=5):
        # Initializes the Skip-Gram model with Negative Sampling parameters and applies Xavier initialization.
        super().__init__(data_path, context_size, learning_rate, hidden_layer_size)

        self.negative_sampling_size = negative_sampling_size

        # Calculate the boundary for Xavier/Glorot initialization to maintain variance across layers.
        limit_w1 = np.sqrt(6 / (self.V + self.N))
        self.W1 = np.random.uniform(-limit_w1, limit_w1, size=(self.V, self.N))
        
        limit_w2 = np.sqrt(6 / (self.V + self.N))
        self.W2 = np.random.uniform(-limit_w2, limit_w2, size=(self.V, self.N))
    
    def update_weights(self, input_vector_id, context_vector_ids):
        # Executes the forward and backward passes for the true context words and generated negative samples.
        
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

        # Update the output vector weights for the true context words.
        np.add.at(self.W2, context_vector_ids, -self.learning_rate * np.outer(gradient_true, h)) # (Eq. 58)

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

        # Update the output vector weights for all selected negative samples.
        np.add.at(self.W2, negative_ids, -self.learning_rate * np.outer(gradient_neg, h)) # (Eq. 58)

        # Update the input-to-hidden weight for the central input word.
        self.W1[input_vector_id] -= self.learning_rate * EH # (Eq. 60)

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