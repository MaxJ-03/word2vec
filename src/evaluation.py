import time
import numpy as np
import os

def get_similar_words(model, target_word, top_n=5):
    # Check if the word actually exists in our vocabulary
    if target_word not in model.data_processing.vocabulary:
        print(f"Word '{target_word}' not found in vocabulary.")
        return

    # Get the integer ID of our target word
    target_id = model.data_processing.word_to_id[target_word]

    # Extract the target word's mathematical vector from W1
    target_vector = model.W1[target_id]

    # Calculate the Cosine Similarity against the entire W1 matrix 
    dot_products = np.dot(model.W1, target_vector)
    norms = np.linalg.norm(model.W1, axis=1) * np.linalg.norm(target_vector)
    
    # Avoid division by zero
    similarities = dot_products / (norms + 1e-9)

    # Sort the indices by highest similarity score
    closest_ids = np.argsort(similarities)[::-1]

    print(f"\nWords most similar to '{target_word}':")
    
    # Explicitly loop and skip the target word
    count = 0
    for idx in closest_ids:
        if idx == target_id:
            continue  # Skip the word itself!
            
        word = model.data_processing.id_to_word[idx]
        score = similarities[idx]
        print(f" - {word} (Score: {score:.4f})")
        
        count += 1
        if count == top_n:
            break

def evaluate_analogies(embeddings_obj, test_file_path):
    """
    Evaluates semantic and syntactic accuracy of word embeddings using the Mikolov analogy test.
    Prints a formatted table to the console and returns a dictionary of the calculated metrics.
    """
    test_filename = os.path.basename(test_file_path)
    print(f"\n--- Running Analogy Benchmark on {test_filename} ---")
    
    start_time = time.time()
    
    vocab = embeddings_obj.vocab if hasattr(embeddings_obj, 'vocab') else embeddings_obj.data_processing.vocabulary
    word_to_id = embeddings_obj.word_to_id if hasattr(embeddings_obj, 'word_to_id') else embeddings_obj.data_processing.word_to_id
    id_to_word = embeddings_obj.id_to_word if hasattr(embeddings_obj, 'id_to_word') else embeddings_obj.data_processing.id_to_word
    W1 = embeddings_obj.W1
    model_name = getattr(embeddings_obj, 'name', 'Training_Session_Model')

    W1_norms = np.linalg.norm(W1, axis=1, keepdims=True)
    W1_norms[W1_norms == 0] = 1e-9
    W1_normalized = W1 / W1_norms

    results = {
        'semantic': {'correct': 0, 'evaluated': 0, 'skipped': 0},
        'syntactic': {'correct': 0, 'evaluated': 0, 'skipped': 0}
    }
    
    current_category = None

    try:
        with open(test_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().lower()
                
                if not line:
                    continue
                    
                if line.startswith(":"):
                    if "gram" in line:
                        current_category = 'syntactic'
                    else:
                        current_category = 'semantic'
                    continue
                
                if current_category is None:
                    continue 

                words = line.split()
                if len(words) != 4:
                    continue
                
                w1, w2, w3, expected_w4 = words
                
                if w1 not in vocab or w2 not in vocab or w3 not in vocab or expected_w4 not in vocab:
                    results[current_category]['skipped'] += 1
                    continue
                
                results[current_category]['evaluated'] += 1

                id1, id2, id3 = word_to_id[w1], word_to_id[w2], word_to_id[w3]
                
                target_vector = W1_normalized[id2] - W1_normalized[id1] + W1_normalized[id3]
                
                similarities = np.dot(W1_normalized, target_vector)
                
                similarities[id1] = -np.inf
                similarities[id2] = -np.inf
                similarities[id3] = -np.inf
                
                predicted_id = np.argmax(similarities)
                predicted_word = id_to_word[predicted_id]
                
                if predicted_word == expected_w4:
                    results[current_category]['correct'] += 1
                    
    except FileNotFoundError:
        print(f"Error: Could not find {test_file_path}")
        return None

    sem_eval = results['semantic']['evaluated']
    syn_eval = results['syntactic']['evaluated']
    tot_eval = sem_eval + syn_eval
    
    sem_cor = results['semantic']['correct']
    syn_cor = results['syntactic']['correct']
    tot_cor = sem_cor + syn_cor

    sem_skip = results['semantic']['skipped']
    syn_skip = results['syntactic']['skipped']
    tot_skip = sem_skip + syn_skip

    sem_acc = (sem_cor / sem_eval * 100) if sem_eval > 0 else 0
    syn_acc = (syn_cor / syn_eval * 100) if syn_eval > 0 else 0
    tot_acc = (tot_cor / tot_eval * 100) if tot_eval > 0 else 0

    eval_time = time.time() - start_time

    print("\n" + "="*55)
    print(f"{'Category':<15} | {'Evaluated':<10} | {'Correct':<8} | {'Accuracy':<8}")
    print("-" * 55)
    print(f"{'Semantic':<15} | {sem_eval:<10} | {sem_cor:<8} | {sem_acc:.2f}%")
    print(f"{'Syntactic':<15} | {syn_eval:<10} | {syn_cor:<8} | {syn_acc:.2f}%")
    print("-" * 55)
    print(f"{'Total':<15} | {tot_eval:<10} | {tot_cor:<8} | {tot_acc:.2f}%")
    print("="*55)
    print(f"Questions Skipped (Missing Vocab): {tot_skip}")
    print(f"Evaluation finished in {eval_time:.2f} seconds\n")

    metrics = {
        'model_name': model_name,
        'semantic_evaluated': sem_eval,
        'semantic_accuracy': sem_acc,
        'syntactic_evaluated': syn_eval,
        'syntactic_accuracy': syn_acc,
        'total_accuracy': tot_acc,
        'skipped': tot_skip
    }
    
    return metrics