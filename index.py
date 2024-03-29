import numpy as np
import pandas as pd
import pickle
import string
import nltk

nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stopwords = nltk.corpus.stopwords.words('english')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

#---------------Functions to clean the input text--------------#

def tokenize_remove_punctuations(text):
    clean_text = []
    text = text.split(" ")
    for word in text:
        word = list(word)
        new_word = []
        for c in word:
            if c not in string.punctuation:
                new_word.append(c)
        word = "".join(new_word)
        if len(word) > 0:
            clean_text.append(word)
    return clean_text
    
def remove_stopwords(text):
    clean_text = []
    for word in text:
        if word not in stopwords:
            clean_text.append(word)
    return clean_text
    
def pos_tagging(text):
    tagged = nltk.pos_tag(text)
    return tagged
            
def get_wordnet(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize(pos_tags):
    lemmatized_text = []
    for t in pos_tags:
        word = WordNetLemmatizer().lemmatize(t[0], get_wordnet(t[1]))
        lemmatized_text.append(word)
    return lemmatized_text
        
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = tokenize_remove_punctuations(text)
    text = [word for word in text if not any(c.isdigit() for c in word)]
    text = remove_stopwords(text)
    text = [t for t in text if len(t) > 0]
    pos_tags = pos_tagging(text)
    text = lemmatize(pos_tags)
    text = [t for t in text if len(t) > 1]
    text = " ".join(text)
    return text

#---------------Loading trained models--------------#

with open("vector.pkl", "rb") as f:
    vect = pickle.load(f)

with open("Naive_Bayes.pkl", "rb") as f:
    MultinomialNB = pickle.load(f)

with open("Logistic_Regression.pkl", "rb") as f:
    LogisticRegression = pickle.load(f)

with open("Linear_SVC.pkl", "rb") as f:
    LinearSVC = pickle.load(f)

#---------------Predicting output--------------#

def predict_comment(comment, option):
    comment = clean_text(comment)
    comment = np.array([comment])
    comment = pd.Series(comment)
    comment = vect.transform(comment)

    if option == 'Multinomial Naive Bayes':
        pred = MultinomialNB.predict(comment)
        return "Bullying comment!!!!" if pred[0] == 1 else "Normal comment."
    elif option == 'Linear SVC':
        pred = LinearSVC.predict(comment)
        return "Bullying comment!!!!" if pred[0] == 1 else "Normal comment."
    elif option == 'Logistic Regression':
        pred = LogisticRegression.predict(comment)
        return "Bullying comment!!!!" if pred[0] == 1 else "Normal comment."
    else:
        return "You haven't selected any model :("


#---------------CLI--------------#

def main():
    print("Cyber Bullying Detector")
    option = input("Select Model (Linear SVC, Logistic Regression, Multinomial Naive Bayes): ")
    comment = input("Enter any comment: ")
    prediction = predict_comment(comment, option)
    print("Prediction:", prediction)

if __name__ == "__main__":
    main()
