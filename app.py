import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")



# import streamlit as st
# import pickle

# import nltk
# nltk.download('punkt_tab')
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# from nltk.tokenize import word_tokenize

# import string

# stopwords = stopwords.words('english')
# ps = PorterStemmer()



# tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
# model = pickle.load(open('model.pkl', 'rb'))

# st.title('Sms Spam Classifier')
# input_text = st.text_area('Enter your sms here')

# def transform_text(text):
#     text = text.lower()  #lower text
#     text = nltk.word_tokenize(text) #tokenize
#     y= []
#     for i in text:  # removing special character
#         if i.isalnum():
#             y.append(i)
#     text = y[:]
#     y.clear()
#     for i in text:
#         if i not in stopwords and i not in string.punctuation:
#             y.append(i)
#     text = y[:]
#     y.clear()
    
#     for i in text:
#       y.append(ps.stem(i))

#     return " ".join(y)
# if st.button('Predict'):
#     transformed_text = transform_text(input_text)
#     vectorized_text = tfidf.transform([transformed_text])
#     prediction = model.predict(vectorized_text)[0]
#     if prediction == 0:
#         st.header('This is a SPAM sms')
#     else:
#         st.header('This is a Not SPAM sms')
        



