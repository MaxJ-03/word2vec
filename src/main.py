import os
import time
import numpy as np
from skip_gram.skip_gram_neg_sample import SkipGramNegativeSampling
from skip_gram.skip_gram import SkipGram
from skip_gram.skip_gram_hier_softmax import SkipGramHierarchical

from cbow.cbow import CBOW
from cbow.cbow_neg_sample import CBOWNegativeSampling
from cbow.cbow_hier_softmax import CBOWHierarchical

from evaluation import get_similar_words


if __name__ == "__main__":
    # 1. Create a dummy dataset for testing
    test_file_path = "../trainsets/special_sauce.txt"

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
    
    # 4. Start the timer
    start_time = time.time()
    
    # 5. Train the model
    model.train(epochs=EPOCHS)

    # 6. Stop the timer and calculate the difference
    end_time = time.time()
    execution_time = end_time - start_time
    
    print("\n--- Training Complete ---")
    print(f"Total Execution Time: {execution_time:.4f} seconds")

    print("\n--- Evaluation ---")
    get_similar_words(model, "pitch")
    get_similar_words(model, "reflection")
    get_similar_words(model, "cv")