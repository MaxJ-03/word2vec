import os
import sys
import time

from cbow.cbow import CBOW
from cbow.cbow_hier_softmax import CBOWHierarchical
from cbow.cbow_neg_sample import CBOWNegativeSampling
from skip_gram.skip_gram import SkipGram
from skip_gram.skip_gram_hier_softmax import SkipGramHierarchical
from skip_gram.skip_gram_neg_sample import SkipGramNegativeSampling

from evaluation import get_similar_words, evaluate_analogies

def get_valid_input(prompt, valid_range=None, type_cast=int, default=None):
    """Helper function to guarantee the user enters a valid number without crashing."""
    while True:
        try:
            user_input = input(prompt).strip()
            # If the user just hits Enter, return the default value
            if not user_input and default is not None:
                return default
            
            val = type_cast(user_input)
            
            # Check if it's within our numbered menu options
            if valid_range and val not in valid_range:
                print(f" -> Please enter a number between {valid_range[0]} and {valid_range[-1]}.")
                continue
            return val
        except ValueError:
            print(" -> Invalid input. Please enter a valid number.")

def get_multiple_inputs(prompt, valid_range):
    """Helper function to allow selecting multiple models via a comma-separated list."""
    while True:
        try:
            user_input = input(prompt).strip()
            if not user_input:
                print(" -> Please enter at least one number.")
                continue
            
            # Split by comma and strip spaces
            parts = [p.strip() for p in user_input.split(',')]
            choices = []
            
            for p in parts:
                val = int(p)
                if valid_range and val not in valid_range:
                    print(f" -> {val} is not a valid option.")
                    raise ValueError()
                choices.append(val)
            
            # Remove duplicates using a dictionary while keeping the original order
            return list(dict.fromkeys(choices))
        except ValueError:
            print(" -> Invalid input. Please enter numbers separated by commas (e.g., 1, 3, 6).")

def run_training():
    print("=" * 60)
    print("   Word2Vec Custom Architecture Training Hub")
    print("=" * 60)

    # Interactive model selection
    models = [
        ("Standard CBOW", CBOW, False),
        ("CBOW with Hierarchical Softmax", CBOWHierarchical, False),
        ("CBOW with Negative Sampling", CBOWNegativeSampling, True),
        ("Standard Skip-Gram", SkipGram, False),
        ("Skip-Gram with Hierarchical Softmax", SkipGramHierarchical, False),
        ("Skip-Gram with Negative Sampling", SkipGramNegativeSampling, True)
    ]

    print("\n[ Select Architecture ]")
    for i, (name, _, _) in enumerate(models, 1):
        print(f"  {i}. {name}")
    print(f"  7. Run All Models Sequentially")
    
    # Ask for multiple choices separated by commas
    choices = get_multiple_inputs("\nEnter model numbers separated by commas (e.g., 3, 6) or 7 for All: ", valid_range=range(1, 8))
    
    # Build the target queue based on the choices
    if 7 in choices:
        models_to_run = models
        is_batch_mode = True
    else:
        models_to_run = [models[c - 1] for c in choices]
        # It's a "Batch" if they selected more than 1 model!
        is_batch_mode = len(models_to_run) > 1 

    # Check if any of the selected models need negative sampling
    uses_negative_sampling = any(m_info[2] for m_info in models_to_run)

    # Interactive dataset selection
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    trainsets_dir = os.path.join(base_dir, 'trainsets')

    if not os.path.exists(trainsets_dir):
        trainsets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trainsets')
        if not os.path.exists(trainsets_dir):
            print("\n[!] Could not find a 'trainsets' folder. Please create one.")
            return

    txt_files = [f for f in os.listdir(trainsets_dir) if f.endswith('.txt')]
    
    if not txt_files:
        print(f"\n[!] No .txt files found in '{trainsets_dir}'. Please add some datasets and try again.")
        return

    print("\n[ Select Dataset ]")
    for i, f_name in enumerate(txt_files, 1):
        print(f"  {i}. {f_name}")
    
    file_choice = get_valid_input("\nEnter the number of your dataset: ", valid_range=range(1, len(txt_files) + 1)) - 1
    selected_file = os.path.join(trainsets_dir, txt_files[file_choice])

    # Interactive Hyperparameter selection
    print("\n[ Hyperparameters ] (Press Enter to use defaults)")
    epochs = get_valid_input("  Epochs (default 50): ", default=50)
    print_interval = get_valid_input("  Printing Interval (default 100000): ", default=100000)
    lr = get_valid_input("  Learning Rate (default 0.05): ", type_cast=float, default=0.05)
    window = get_valid_input("  Context Window Size (default 2): ", default=2)
    dim = get_valid_input("  Embedding Dimension (default 10): ", default=10)
    
    neg_samples = 5
    if uses_negative_sampling:
        neg_samples = get_valid_input("  Negative Samples (default 5): ", default=5)

    # Automatically locate the analogy test file in the 'testsets' folder
    testsets_dir = os.path.join(base_dir, 'testsets')
    auto_test_path = os.path.join(testsets_dir, 'word-test.v1.txt')
    
    if not os.path.exists(auto_test_path):
        auto_test_path = 'testsets/word-test.v1.txt'

    # Initialization and Training
    print("\n" + "=" * 60)
    if is_batch_mode:
        print(f" BATCH MODE INITIATED: Training queue of {len(models_to_run)} architectures")
    else:
        print(f" INITIATING TRAINING: {models_to_run[0][0]}")
    print("=" * 60)

    global_start_time = time.time()
    last_trained_model = None

    # Loop through our target queue
    for i, (model_name, ModelClass, requires_neg_sampling) in enumerate(models_to_run, 1):
        print(f"\n--- [{i}/{len(models_to_run)}] Initializing {model_name} ---")
        
        if requires_neg_sampling:
            model = ModelClass(selected_file, window, lr, dim, negative_sampling_size=neg_samples)
        else:
            model = ModelClass(selected_file, window, lr, dim)

        print("\nStarting Training Loop...")

        main_start_time = time.time()
        model.train(epochs, print_interval)
        main_end_time = time.time()
        
        total_time = main_end_time - main_start_time
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)
        
        print("\n" + "="*40)
        print(f"Training Complete for {model_name} in {int(hours)}h {int(minutes)}m {seconds:.2f}s!")
        print("="*40)

        # Save the embeddings
        clean_dataset_name = os.path.splitext(os.path.basename(selected_file))[0]
        safe_model_name = model_name.replace(" ", "_")
        
        # Pass the epochs, and pass the neg_samples only if the model uses them
        if requires_neg_sampling:
            model.save_embeddings(clean_dataset_name, safe_model_name, epochs, neg_samples)
        else:
            model.save_embeddings(clean_dataset_name, safe_model_name, epochs)

        # Perform analogy evaluation
        print("\n" + "-"*40)
        print(f"Running Automatic Analogy Test for {model_name}...")
        if os.path.exists(auto_test_path):
            evaluate_analogies(model, auto_test_path)
        else:
            print(f"Could not find '{auto_test_path}'. Please ensure 'word-test.v1.txt' is inside the 'testsets' folder.")

        last_trained_model = model

    # Calculate Total Time
    global_end_time = time.time()
    global_total = global_end_time - global_start_time
    ghours, grem = divmod(global_total, 3600)
    gminutes, gseconds = divmod(grem, 60)
    print("\n" + "="*60)
    print(f"All tasks complete! Total run time: {int(ghours)}h {int(gminutes)}m {gseconds:.2f}s")
    print("="*60)

    # Interactive Evaluation loop (Skip if running multiple models)
    if not is_batch_mode and last_trained_model is not None:
        print("-" * 60)
        
        top_5_tuples = last_trained_model.data_processing.vocabulary_frequency.most_common(5)
        suggestions = [word for word, count in top_5_tuples]
        
        print("Interactive Evaluation ready! You can:")
        print(" 1. Type a word to see its closest neighbors.")
        print(f"Word suggestions: {', '.join(suggestions)}")
        
        while True:
            user_input = input("\nEnter word (or press Enter to quit): ").strip().lower()
            
            if not user_input:
                print("Exiting hub. Have a great day!")
                break
                
            get_similar_words(last_trained_model, user_input, top_n=5)
    
    elif is_batch_mode:
        print("\nNote: Interactive evaluation is skipped in Batch Mode.")
        print("Your embeddings are saved in the 'embeddings/' folder.")

if __name__ == '__main__':
    run_training()