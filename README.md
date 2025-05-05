# Advanced NLP - Homework 1: Sentiment Analysis with Rotten Tomatoes

This repository contains the solution for Homework 1 of the Advanced Methods in NLP course.

## Repository Structure

### Main Notebook
- `hw1_sentiment_analysis_RT.ipynb`: Jupyter notebook containing the complete solutions to all homework questions.

### Library Package
- `hw1_library/`: A Python package containing classes and utility functions used in the notebook:
  - `preprocessing/`: Text preprocessing utilities (stemming, lemmatization, etc.)
  - `metrics/`: Evaluation metrics implementation
  - `regex_classifiers/`: Rule-based classifiers using regular expressions
  - `utilities/`: Helper functions for analysis and visualization

### Models
- `models/`: Directory containing trained models saved during notebook execution:
  - Logistic regression models with various feature extraction methods
  - LSTM models with Word2Vec embeddings (both fixed and fine-tuned)

## Dataset

The homework uses the Rotten Tomatoes dataset for sentiment analysis of movie reviews.

## Author

Pablo Fernández