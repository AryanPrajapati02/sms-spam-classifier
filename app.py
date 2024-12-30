

import streamlit as st
import pickle

import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import string

stopwords = stopwords.words('english')
ps = PorterStemmer()



tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Sms Spam Classifier')
input_text = st.text_area('Enter your sms here')

def transform_text(text):
    text = text.lower()  #lower text
    text = nltk.word_tokenize(text) #tokenize
    y= []
    for i in text:  # removing special character
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    
    for i in text:
      y.append(ps.stem(i))

    return " ".join(y)
if st.button('Predict'):
    transformed_text = transform_text(input_text)
    vectorized_text = tfidf.transform([transformed_text])
    prediction = model.predict(vectorized_text)[0]
    if prediction == 1:
        st.header('This is a SPAM sms')
    else:
        st.header('This is a Not SPAM sms')
        



