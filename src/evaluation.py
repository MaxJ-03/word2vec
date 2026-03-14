import numpy as np

def get_similar_words(model, target_word, top_n=5):
    # 1. Check if the word actually exists in our vocabulary
    if target_word not in model.data_processing.vocabulary:
        print(f"Word '{target_word}' not found in vocabulary.")
        return

    # 2. Get the integer ID of our target word
    target_id = model.data_processing.word_to_id[target_word]

    # 3. Extract the target word's mathematical vector from W1
    target_vector = model.W1[target_id]

    # 4. Calculate the Cosine Similarity against the entire W1 matrix instantly!
    dot_products = np.dot(model.W1, target_vector)
    norms = np.linalg.norm(model.W1, axis=1) * np.linalg.norm(target_vector)
    
    # Avoid division by zero just in case
    similarities = dot_products / (norms + 1e-9)

    # 5. Sort the indices by highest similarity score
    # argsort sorts lowest to highest, so [::-1] flips it to highest first
    closest_ids = np.argsort(similarities)[::-1]

    print(f"\nWords most similar to '{target_word}':")
    
    # 6. Loop through the top N closest words (skipping the 1st one, which is the word itself!)
    for idx in closest_ids[1:top_n + 1]:
        word = model.data_processing.id_to_word[idx]
        score = similarities[idx]
        print(f" - {word} (Score: {score:.4f})")