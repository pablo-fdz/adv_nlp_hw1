import numpy as np

# Calculate overall statistical changes in the embedding space
def analyze_embedding_changes(original, tuned, word_index, vocab_size):

    """
    Analyze changes in word embeddings before and after tuning.

    Parameters:
    - original (np.ndarray): Original word embeddings.
    - tuned (np.ndarray): Tuned word embeddings.
    - word_index (dict): Mapping from words to their indices in the embeddings.
    - vocab_size (int): Size of the vocabulary.

    Returns:
    - dict: A dictionary containing:
        - 'similarities': List of cosine similarities for each word.
        - 'distances': List of Euclidean distances for each word.
        - 'relative_changes': List of relative changes for each word.
        - 'valid_count': Number of words with valid embeddings.
    """

    # Words that actually have embeddings (non-zero vectors)
    valid_indices = []
    
    for word, idx in word_index.items():
        if idx < vocab_size and np.any(original[idx]):
            valid_indices.append(idx)
    
    # Calculate similarities and distances for all valid word vectors
    similarities = []
    distances = []
    relative_changes = []
    
    for idx in valid_indices:
        orig_vec = original[idx]
        tuned_vec = tuned[idx]
        
        orig_norm = np.linalg.norm(orig_vec)
        tuned_norm = np.linalg.norm(tuned_vec)
        
        # Compute cosine similarity
        similarity = np.dot(orig_vec, tuned_vec) / (orig_norm * tuned_norm)
        similarities.append(similarity)
        
        # Compute Euclidean distance
        distance = np.linalg.norm(orig_vec - tuned_vec)
        distances.append(distance)
        
        # Compute relative change
        relative_change = distance / orig_norm
        relative_changes.append(relative_change)
    
    return {
        'similarities': similarities,
        'distances': distances,
        'relative_changes': relative_changes,
        'valid_count': len(valid_indices)
    }