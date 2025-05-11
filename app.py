import streamlit as st
import pickle
import unicodedata
import re
import string
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

st.set_page_config(page_title="Truyen Kieu Search")

# Preprocessing

def tokenize(text):
    text = unicodedata.normalize('NFC', text.lower())
    tokens = text.split()
    return [t.strip(string.punctuation) for t in tokens if t.strip(string.punctuation)]

def preprocess_text(text):
    text = unicodedata.normalize('NFC', text.lower())
    return re.sub(r'\s+', ' ', text).strip()

# Load models
@st.cache_resource
def load_model():
    with open("kieu_model.pkl", "rb") as f:
        data = pickle.load(f)
    return data["model"], data["vectorizer"], data["verses"], data["search_vectorizer"], data["search_doc_matrix"]

model, clf_vectorizer, verses, search_vectorizer, search_doc_matrix = load_model()

# Inverted index
@st.cache_resource
def build_inverted_index(verses):
    index = defaultdict(list)
    for i, verse in enumerate(verses):
        for word in set(tokenize(verse)):
            index[word].append(i)
    return index

inverted_index = build_inverted_index(verses)

# UI
st.title("Truyen Kieu Search & Authorship Prediction")
tab1, tab2, tab3, tab4 = st.tabs(["Classify Verse", "Search by Cosine", "Search by Overlap", "Inverted Index"])

with tab1:
    verse_input = st.text_input("Input a verse to predict the author")
    if verse_input:
        processed = preprocess_text(verse_input)
        features = clf_vectorizer.transform([processed])
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][pred]
        label = "Nguyen Du" if pred == 1 else "Other Author"
        st.write("Prediction:", label)
        st.write("Probability:", f"{prob:.2f}")

with tab2:
    query = st.text_input("Search by cosine similarity")
    if query:
        query_vec = search_vectorizer.transform([preprocess_text(query)])
        sims = cosine_similarity(query_vec, search_doc_matrix).flatten()
        top_ids = sims.argsort()[-5:][::-1]
        st.write("Top matching verses:")
        for i in top_ids:
            st.write(f"- {verses[i]} (score: {sims[i]:.2f})")

with tab3:
    overlap_query = st.text_input("Search by word overlap")
    if overlap_query:
        query_tokens = set(tokenize(overlap_query))
        scores = [(i, len(query_tokens & set(tokenize(v)))) for i, v in enumerate(verses)]
        top = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
        st.write("Top matching verses:")
        for i, score in top:
            st.write(f"- {verses[i]} (overlap: {score})")

with tab4:
    keyword = st.text_input("Find all verses containing a word")
    if keyword:
        word = keyword.strip().lower()
        results = inverted_index.get(word, [])
        if results:
            st.write(f"Found in {len(results)} verse(s):")
            for idx in results[:10]:
                st.write(f"- {verses[idx]}")
        else:
            st.write("Word not found.")
