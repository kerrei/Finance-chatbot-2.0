import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load terms
with open("financial_terms.txt", "r") as f:
    terms_list = [line.strip() for line in f if line.strip()]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(terms_list)

def chatbot_response(query):
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, X).flatten()
    idx = similarity.argmax()
    return terms_list[idx]

st.title("💬 Financial Terms Chatbot")
user_input = st.text_input("Ask about a financial term:")
if user_input:
    response = chatbot_response(user_input)
    st.write("**Closest Match:**", response)
