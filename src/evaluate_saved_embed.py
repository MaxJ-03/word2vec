import os
import json
import numpy as np
from evaluation import evaluate_analogies

class LoadedEmbeddings:
    def __init__(self, filepath):
        self.name = os.path.basename(filepath).replace('.txt', '')
        self.word_to_id = {}
        self.id_to_word = {}
        self.W1 = None
        self.vocab = set()
        self.load(filepath)

    def load(self, filepath):
        """
        Parses the saved embedding file and loads the vectors into memory.
        """
        print(f"\nLoading embeddings from {self.name}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        vocab_size, dim = map(int, lines[0].strip().split())
        self.W1 = np.zeros((vocab_size, dim))
        
        for i, line in enumerate(lines[1:]):
            parts = line.strip().split()
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]])
            
            self.word_to_id[word] = i
            self.id_to_word[i] = word
            self.W1[i] = vector
            self.vocab.add(word)
            
        print(f"Successfully loaded {vocab_size} words into {dim}-dimensional space.")

    def get_similar_words(self, target_word, top_n=5):
        """
        Computes cosine similarity to find the nearest neighbors for a given target word.
        """
        print(f"\n[{self.name}]")
        if target_word not in self.vocab:
            print(f" -> Word '{target_word}' not found in vocabulary.")
            return

        target_id = self.word_to_id[target_word]
        target_vector = self.W1[target_id]

        dot_products = np.dot(self.W1, target_vector)
        norms = np.linalg.norm(self.W1, axis=1) * np.linalg.norm(target_vector)
        similarities = dot_products / (norms + 1e-9)

        closest_ids = np.argsort(similarities)[::-1]

        count = 0
        for idx in closest_ids:
            if idx == target_id:
                continue
            word = self.id_to_word[idx]
            score = similarities[idx]
            print(f"  - {word} (Score: {score:.4f})")
            count += 1
            if count == top_n:
                break

def get_multiple_inputs(prompt, valid_range):
    """
    Parses a comma-separated string of user inputs into a deduplicated list of integers.
    """
    while True:
        try:
            user_input = input(prompt).strip()
            if not user_input:
                print(" -> Please enter at least one number.")
                continue
            
            parts = [p.strip() for p in user_input.split(',')]
            choices = []
            
            for p in parts:
                val = int(p)
                if valid_range and val not in valid_range:
                    print(f" -> {val} is not a valid option.")
                    raise ValueError()
                choices.append(val)
            
            return list(dict.fromkeys(choices))
        except ValueError:
            print(" -> Invalid input. Please enter numbers separated by commas.")

def save_metrics_to_log(metrics, base_dir):
    """
    Appends evaluation metrics to a persistent JSON log file.
    """
    if metrics is None:
        return

    reports_dir = os.path.join(base_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    log_path = os.path.join(reports_dir, 'metrics_log.json')

    log_data = []
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            try:
                log_data = json.load(f)
            except json.JSONDecodeError:
                log_data = []

    # Filter out previous runs of the same model to keep the log clean
    log_data = [entry for entry in log_data if entry['model_name'] != metrics['model_name']]
    log_data.append(metrics)

    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=4)

def run_evaluation():
    """
    Launches the interactive evaluation interface for loaded models.
    """
    print("=" * 60)
    print("   Word2Vec Multi-Model Evaluation Hub")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    embeddings_dir = os.path.join(base_dir, 'embeddings')
    testsets_dir = os.path.join(base_dir, 'testsets')
    
    if not os.path.exists(embeddings_dir):
        print("\n[!] Could not find 'embeddings' folder. Have you trained a model yet?")
        return

    files = [f for f in os.listdir(embeddings_dir) if f.endswith('.txt')]
    if not files:
        print("\n[!] No saved embeddings found in the 'embeddings' folder.")
        return

    print("\n[ Saved Embedding Files ]")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f}")
        
    choices = get_multiple_inputs("\nEnter file numbers separated by commas to load (e.g., 1, 2): ", valid_range=range(1, len(files) + 1))
    
    loaded_models = []
    for choice in choices:
        selected_file = os.path.join(embeddings_dir, files[choice - 1])
        model = LoadedEmbeddings(selected_file)
        loaded_models.append(model)
    
    print("\n" + "=" * 60)
    print(" Evaluation Ready. Comparing models side-by-side.")
    print(" 1. Type a word to see its closest neighbors.")
    print(" 2. Type 'TEST' to run an analogy benchmark.")
    print("=" * 60)
    
    while True:
        user_input = input("\nEnter word (or 'TEST', or press Enter to quit): ").strip().lower()
        if not user_input:
            print("Returning to Main Menu.")
            break
            
        if user_input == 'test':
            if not os.path.exists(testsets_dir):
                print("\n[!] Could not find 'testsets' folder.")
                continue
                
            test_files = [f for f in os.listdir(testsets_dir) if f.endswith('.txt')]
            if not test_files:
                print("\n[!] No .txt files found in 'testsets' folder.")
                continue
                
            print("\n[ Available Test Sets ]")
            for i, f_name in enumerate(test_files, 1):
                print(f"  {i}. {f_name}")
                
            test_choices = get_multiple_inputs("\nEnter test set numbers separated by commas (e.g., 1): ", valid_range=range(1, len(test_files) + 1))
            
            for tc in test_choices:
                test_filename = test_files[tc - 1]
                test_path = os.path.join(testsets_dir, test_filename)
                
                for model in loaded_models:
                    metrics = evaluate_analogies(model, test_path)
                    
                    # Prevent crash if the evaluation failed
                    if metrics:
                        # save_metrics_to_log should be imported or defined in this file
                        save_metrics_to_log(metrics, base_dir)
                    
            print("\nMetrics successfully saved to reports/metrics_log.json")
        else:
            for model in loaded_models:
                model.get_similar_words(user_input)

if __name__ == '__main__':
    run_evaluation()