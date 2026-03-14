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