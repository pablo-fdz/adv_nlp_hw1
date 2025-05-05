from .lower_nostop import lower_nostop
import numpy as np

def word2vec_tfidf_weighted_embedding(texts, model, tfidf_vectorizer, rm_stopwords=True, stopwords=None):
    
    """
    Compute the mean word2vec embedding for a given text. Does not consider words not in the model.

    Parameters:
    - texts: pandas Series, containing the text strings to be processed.
    - model: Word2Vec model, the pre-trained word2vec model.
    - tfidf_vectorizer: fitted TfidfVectorizer, used to get word weights.
    - rm_stopwords: bool, whether to remove stopwords or not.
    - stopwords: set, a set of stopwords to be removed if rm_stopwords is True.

    Returns:
    - np.ndarray, an array with the TF-IDF-weighted word2vec embedding for each input text in the pandas series.
    """

    features = []
    
    # Get feature names from vectorizer
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Transform all texts to get the TF-IDF matrix
    processed_texts = [lower_nostop(text, rm_stopwords=rm_stopwords, 
                                   stopword_set=stopwords, output_type='string') 
                      for text in texts.values]
    tfidf_matrix = tfidf_vectorizer.transform(processed_texts)
    
    for i, text in enumerate(texts.values):
        
        # Ensure text is a string before processing
        if not isinstance(text, str):
            raise TypeError(f"Expected a string, got {type(text)}")
        
        # Get preprocessed words
        words = lower_nostop(text, rm_stopwords=rm_stopwords, 
                            stopword_set=stopwords, output_type="list")
        
        # Get word vectors and their weights
        weighted_vectors = []
        total_weight = 0
        
        # Get non-zero elements in this document's TF-IDF vector
        tfidf_indices = tfidf_matrix[i].nonzero()[1]
        tfidf_weights = {}
        
        # Map indices to words and weights
        for idx in tfidf_indices:
            word = feature_names[idx]
            weight = tfidf_matrix[i, idx]
            tfidf_weights[word] = weight
        
        # Create weighted vectors
        for word in words:
            if word in model and word in tfidf_weights:
                weight = tfidf_weights[word]
                weighted_vectors.append(model[word] * weight)
                total_weight += weight
        
        # Handle case where no words are in the model or have TF-IDF weights
        if len(weighted_vectors) == 0 or total_weight == 0:
            features.append(np.zeros(model.vector_size))
        else:
            # Compute weighted average
            weighted_avg = np.sum(weighted_vectors, axis=0) / total_weight
            features.append(weighted_avg)
    
    return np.vstack(features)