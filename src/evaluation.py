import numpy as np

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

def evaluate_analogies(model, test_file_path):
    print(f"\n--- Running Analogy Evaluation on {test_file_path} ---")
    
    vocab = model.data_processing.vocabulary
    word_to_id = model.data_processing.word_to_id
    id_to_word = model.data_processing.id_to_word
    W1 = model.W1

    correct = 0
    total_evaluated = 0
    total_skipped = 0

    try:
        with open(test_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().lower()
                
                # Skip category headers (e.g., ": capital-common-countries")
                if line.startswith(":") or not line:
                    continue
                
                words = line.split()
                if len(words) != 4:
                    continue
                
                w1, w2, w3, expected_w4 = words
                
                # We can only test this if ALL 4 words are in our trained vocabulary
                if w1 not in vocab or w2 not in vocab or w3 not in vocab or expected_w4 not in vocab:
                    total_skipped += 1
                    continue
                
                total_evaluated += 1

                # Get the IDs of the words
                id1, id2, id3 = word_to_id[w1], word_to_id[w2], word_to_id[w3]
                
                # Vector(B) - Vector(A) + Vector(C)
                target_vector = W1[id2] - W1[id1] + W1[id3]
                
                # Calculate Cosine Similarities against the entire vocabulary
                dot_products = np.dot(W1, target_vector)
                norms = np.linalg.norm(W1, axis=1) * np.linalg.norm(target_vector)
                similarities = dot_products / (norms + 1e-9)
                
                # Sort indices from highest similarity to lowest
                closest_ids = np.argsort(similarities)[::-1]
                
                # Find the top guess that isn't one of the input words (w1, w2, w3)
                predicted_word = None
                for idx in closest_ids:
                    if idx not in [id1, id2, id3]:
                        predicted_word = id_to_word[idx]
                        break
                
                if predicted_word == expected_w4:
                    correct += 1
                    
    except FileNotFoundError:
        print(f"Error: Could not find the test file at {test_file_path}")
        return

    print("\n--- Analogy Test Results ---")
    print(f"Total Questions Evaluated: {total_evaluated}")
    print(f"Total Skipped (Missing Vocab): {total_skipped}")
    
    if total_evaluated > 0:
        accuracy = (correct / total_evaluated) * 100
        print(f"Correct Answers: {correct}")
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("Accuracy: N/A (No questions evaluated due to vocabulary mismatch)")