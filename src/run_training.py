import os
import sys
import time
import json

from cbow.cbow import CBOW
from cbow.cbow_hier_softmax import CBOWHierarchical
from cbow.cbow_neg_sample import CBOWNegativeSampling
from skip_gram.skip_gram import SkipGram
from skip_gram.skip_gram_hier_softmax import SkipGramHierarchical
from skip_gram.skip_gram_neg_sample import SkipGramNegativeSampling

from evaluation import evaluate_analogies

def train_model(dataset_path, architectures, epochs, lr, dim, window, ns):
    """
    Core training engine that preserves the complete metrics dictionary 
    format from the evaluation module.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(base_dir, 'reports')
    testsets_dir = os.path.join(base_dir, 'testsets')
    auto_test_path = os.path.join(testsets_dir, 'word-test.v1.txt')
    
    os.makedirs(reports_dir, exist_ok=True)
    log_path = os.path.join(reports_dir, 'metrics_log.json')
    
    model_map = {
        "Standard_CBOW": (CBOW, False),
        "CBOW_with_Hierarchical_Softmax": (CBOWHierarchical, False),
        "CBOW_with_Negative_Sampling": (CBOWNegativeSampling, True),
        "Standard_Skip-Gram": (SkipGram, False),
        "Skip-Gram_with_Hierarchical_Softmax": (SkipGramHierarchical, False),
        "Skip-Gram_with_Negative_Sampling": (SkipGramNegativeSampling, True)
    }

    for arch_key in architectures:
        if arch_key not in model_map:
            continue
            
        ModelClass, requires_ns = model_map[arch_key]
        print(f"\n--- Initializing {arch_key.replace('_', ' ')} ---")
        
        if requires_ns:
            model = ModelClass(dataset_path, window, lr, dim, negative_sampling_size=ns)
        else:
            model = ModelClass(dataset_path, window, lr, dim)

        model.train(epochs, print_interval=1000000)
        
        dataset_basename = os.path.splitext(os.path.basename(dataset_path))[0]
        if requires_ns:
            model.save_embeddings(dataset_basename, arch_key, epochs, ns)
            actual_filename = f"{dataset_basename}_{arch_key}_ep{epochs}_lr{lr}_dim{dim}_w{window}_ns{ns}.txt"
        else:
            model.save_embeddings(dataset_basename, arch_key, epochs)
            actual_filename = f"{dataset_basename}_{arch_key}_ep{epochs}_lr{lr}_dim{dim}_w{window}.txt"

        # Obtain the full dictionary from evaluation.py
        full_metrics = {}
        if os.path.exists(auto_test_path):
            full_metrics = evaluate_analogies(model, auto_test_path)
            
        # Ensure the dictionary uses the actual filename for the leaderboard logic
        # while keeping all other evaluation data (skipped, evaluated counts, etc.)
        if isinstance(full_metrics, dict):
            full_metrics['model_name'] = actual_filename
        else:
            # Fallback if evaluation failed or file was missing
            full_metrics = {
                "model_name": actual_filename,
                "semantic_evaluated": 0,
                "semantic_accuracy": 0.0,
                "syntactic_evaluated": 0,
                "syntactic_accuracy": 0.0,
                "total_accuracy": 0.0,
                "skipped": 0
            }

        log_data = []
        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8') as f:
                try:
                    log_data = json.load(f)
                except json.JSONDecodeError:
                    log_data = []
        
        log_data.append(full_metrics)
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=4)
            
        print(f"Logged full metrics for {actual_filename}")

    return True

def get_valid_input(prompt, valid_range=None, type_cast=int, default=None):
    while True:
        try:
            user_input = input(prompt).strip()
            if not user_input and default is not None:
                return default
            val = type_cast(user_input)
            if valid_range and val not in valid_range:
                continue
            return val
        except ValueError:
            continue

def get_multiple_inputs(prompt, valid_range):
    while True:
        try:
            user_input = input(prompt).strip()
            if not user_input:
                continue
            parts = [p.strip() for p in user_input.split(',')]
            choices = [int(p) for p in parts if int(p) in valid_range]
            return list(dict.fromkeys(choices))
        except ValueError:
            continue

def run_training():
    print("=" * 60)
    print(" Word2Vec Architecture Training Hub")
    print("=" * 60)

    arch_options = [
        "Standard_CBOW", "CBOW_with_Hierarchical_Softmax", "CBOW_with_Negative_Sampling",
        "Standard_Skip-Gram", "Skip-Gram_with_Hierarchical_Softmax", "Skip-Gram_with_Negative_Sampling"
    ]

    print("\n[ Select Architecture ]")
    for i, name in enumerate(arch_options, 1):
        print(f"  {i}. {name.replace('_', ' ')}")
    print("  7. All Architectures")

    choices = get_multiple_inputs("\nSelection: ", range(1, 8))
    selected_keys = arch_options if 7 in choices else [arch_options[c-1] for c in choices]

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    trainsets_dir = os.path.join(base_dir, 'trainsets')
    txt_files = [f for f in os.listdir(trainsets_dir) if f.endswith('.txt')]
    
    print("\n[ Select Dataset ]")
    for i, f in enumerate(txt_files, 1):
        print(f"  {i}. {f}")
    
    f_choice = get_valid_input("\nDataset index: ", range(1, len(txt_files) + 1)) - 1
    selected_path = os.path.join(trainsets_dir, txt_files[f_choice])

    print("\n[ Hyperparameters ]")
    epochs = get_valid_input("  Epochs (100): ", default=100)
    lr = get_valid_input("  Learning Rate (0.025): ", type_cast=float, default=0.025)
    window = get_valid_input("  Window (5): ", default=5)
    dim = get_valid_input("  Dimension (50): ", default=50)
    ns = get_valid_input("  Negative Samples (5): ", default=5)

    train_model(selected_path, selected_keys, epochs, lr, dim, window, ns)

if __name__ == '__main__':
    run_training()