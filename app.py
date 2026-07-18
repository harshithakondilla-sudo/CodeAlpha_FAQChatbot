import streamlit as st
from faq import faqs

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import string

stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words and word not in string.punctuation]
    return " ".join(words)

questions = [faq["question"] for faq in faqs]
processed_questions = [preprocess(q) for q in questions]

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(processed_questions)

def chatbot_response(user_input):
    processed_input = preprocess(user_input)
    input_vector = vectorizer.transform([processed_input])

    similarity = cosine_similarity(input_vector, question_vectors)

    best_match = similarity.argmax()
    score = similarity[0][best_match]

    if score > 0.2:
        return faqs[best_match]["answer"]
    else:
        return "Sorry, I don't know the answer."

st.title("FAQ Chatbot")

user_question = st.text_input("Ask a question")

if st.button("Submit"):
    if user_question:
        answer = chatbot_response(user_question)
        st.success(answer)