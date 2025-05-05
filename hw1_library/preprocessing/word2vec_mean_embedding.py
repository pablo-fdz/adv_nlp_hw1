from .lower_nostop import lower_nostop
import numpy as np

def word2vec_mean_embedding(texts, model, rm_stopwords=True, stopwords=None):
    
    """
    Compute the mean word2vec embedding for a given text. Does not consider words not in the model.

    Parameters:
    - texts: pandas Series, containing the text strings to be processed.
    - model: Word2Vec model, the pre-trained word2vec model.
    - rm_stopwords: bool, whether to remove stopwords or not.
    - stopwords: set, a set of stopwords to be removed if rm_stopwords is True.

    Returns:
    - np.ndarray, an array with the mean word2vec embedding for each input text in the pandas series.
    """

    features = []

    for text in texts.values:

        # Ensure text is a string before processing
        if not isinstance(text, str):
            raise TypeError(f"Expected a string, got {type(text)}")

        # Preprocess the text (similar to your existing preprocessing) and tokenize
        words = lower_nostop(text, rm_stopwords=rm_stopwords, stopword_set=stopwords, output_type="list")
        
        # Get word vector for the input string and average them
        word_vectors = [model[word] for word in words if word in model]  # Each word in the input text is replaced by its embedding. The output here is a list of arrays (one per word).
        if len(word_vectors) == 0:  # No words in the model
            features.append(np.zeros(model.vector_size))  # Return a zero vector of the same size as the word vectors  
        else:
            features.append(np.mean(word_vectors, axis=0))  # Return the mean vector along the first axis (i.e., the mean of the word vectors)
    
    return np.vstack(features)