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