Cyberbullying Detection System

A machine learning-based text classification tool designed to automatically identify cyberbullying in user comments. This project applies Natural Language Processing (NLP) and machine learning to promote healthier online interactions.

## Project Overview

The internet can be a hostile place. This tool processes raw text from user comments, cleans and analyzes it, and classifies it as either **harmful (cyberbullying)** or **normal**. This can be used as a moderation tool to flag offensive content for review.

## Technical Details

### How It Works
1.  **Text Preprocessing:** Input text is cleaned and prepared using standard NLP techniques.
    - **Tokenization:** Splitting text into individual words or tokens.
    - **Stopword Removal:** Filtering out common but insignificant words (e.g., "the", "a", "is").
    - **Part-of-Speech (POS) Tagging:** Labeling words based on their grammatical function.
    - **Lemmatization:** Reducing words to their base or dictionary form (e.g., "running" -> "run").

2.  **Feature Extraction:** The preprocessed text is converted into a numerical format (TF-IDF vectors) that machine learning models can understand.

3.  **Classification:** The numerical features are fed into a trained model to predict the category of the comment.
    - Models Implemented: `Multinomial Naive Bayes`, `Logistic Regression`, `Linear Support Vector Classifier (Linear SVC)`.

### Performance
The project compares the performance of the three models based on standard evaluation metrics like **Accuracy, Precision, Recall, and F1-Score**. (You can add your best model's scores here if you have them, e.g., "Linear SVC achieved an F1-score of 92%").

### Technologies Used
*   **Programming Language:** Python
*   **Libraries:**
    - `pandas` & `numpy` for data manipulation
    - `nltk` for natural language processing
    - `scikit-learn` for machine learning and model evaluation
    - `pickle` for model serialization (saving and loading the model)
