import os
import numpy as np
from skipGram import SkipGram
from CBOW import CBOW
from functions import Functions

if __name__ == "__main__":
    # 1. Create a dummy dataset for testing
    test_file_path = "../trainsets/dummy.txt"

    # 2. Set your hyperparameters
    CONTEXT_SIZE = 4
    LEARNING_RATE = 0.025
    HIDDEN_LAYER_SIZE = 80  # This is the 'N' dimension (vector size)
    EPOCHS = 50

    # 3. Instantiate the model
    model = CBOW(
        data_path=test_file_path, 
        context_size=CONTEXT_SIZE, 
        learning_rate=LEARNING_RATE, 
        hidden_layer_size=HIDDEN_LAYER_SIZE
    )

    
    print("Starting training...")
    
    # 4. Train the model
    model.train(epochs=EPOCHS)