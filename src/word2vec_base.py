import os
from data_proc import DataProcessing

class Word2VecBase:
    def __init__(self, data_path, context_size, learning_rate, hidden_layer_size):
        self.context_size = context_size
        self.initial_lr = learning_rate
        self.learning_rate = learning_rate
        self.N = hidden_layer_size
        
        # Load and process data universally
        self.data_processing = DataProcessing()
        self.data_processing.load(data_path)
        
        self.V = len(self.data_processing.vocabulary)
        self.total_training_words = self.data_processing.data_length

    def update_learning_rate(self, current_epoch, total_epochs):
        """
        Epoch-Level Exponential Decay:
        Updates the learning rate once per epoch. 
        Calculates a decay factor to reach 1% of the initial LR by the final epoch.
        """
        decay_factor = 0.01 ** (1 / total_epochs)
        self.learning_rate = max(self.initial_lr * (decay_factor ** current_epoch), 0.0001)

    def save_embeddings(self, dataset_name, model_name, epochs, neg_samples=None):
        """Saves the W1 matrix in the standard universal Word2Vec .txt format."""
        save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'embeddings')
        os.makedirs(save_dir, exist_ok=True)

        # Build a comprehensive filename with all hyperparameters
        ns_str = f"_ns{neg_samples}" if neg_samples is not None else ""
        
        file_name = f"{dataset_name}_{model_name}_ep{epochs}_lr{self.initial_lr}_dim{self.N}_w{self.context_size}{ns_str}.txt"
        save_path = os.path.join(save_dir, file_name)

        print(f"\nSaving {self.V} word embeddings to {save_path}...")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f"{self.V} {self.N}\n")
            for word, word_id in self.data_processing.word_to_id.items():
                vector = self.W1[word_id]
                vec_str = " ".join(f"{val:.6f}" for val in vector)
                f.write(f"{word} {vec_str}\n")
                
        print("Embeddings saved!")