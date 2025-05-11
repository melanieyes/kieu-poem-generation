import streamlit as st
import pickle
import unicodedata
import re
import string
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Set custom page configuration
st.set_page_config(
    page_title="Truyện Kiều Search & Authorship Prediction",
    page_icon="🌸",
    layout="centered"
)

# --- Inject CSS to hide sidebar and tighten layout ---
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Display resized banner image ---
img = Image.open("truyenkieu.jpg")
st.image(img, width=600)  # Adjust width if needed (600–1000 range works well)

# --- Preprocessing functions ---
def tokenize(text):
    text = unicodedata.normalize('NFC', text.lower())
    tokens = text.split()
    return [t.strip(string.punctuation) for t in tokens if t.strip(string.punctuation)]

def preprocess_text(text):
    text = unicodedata.normalize('NFC', text.lower())
    return re.sub(r'\s+', ' ', text).strip()

# --- Load Model & Data ---
@st.cache_resource
def load_model():
    try:
        with open("kieu_model.pkl", "rb") as f:
            data = pickle.load(f)
        return data["model"], data["vectorizer"], data["verses"], data["search_vectorizer"], data["search_doc_matrix"]
    except FileNotFoundError:
        st.error("Model file not found. Please make sure `kieu_model.pkl` is in the project directory.")
        st.stop()

model, clf_vectorizer, verses, search_vectorizer, search_doc_matrix = load_model()

# --- Build Inverted Index ---
@st.cache_resource
def build_inverted_index(verses):
    index = defaultdict(list)
    for i, verse in enumerate(verses):
        for word in set(tokenize(verse)):
            index[word].append(i)
    return index

inverted_index = build_inverted_index(verses)

# --- Title and Tabs ---
st.title("✨ Truyện Kiều Verse Search & Authorship Classifier")

tab1, tab2, tab3, tab4 = st.tabs([
    "🧠 Classify Verse",
    "📐 Search by Cosine",
    "🔤 Search by Overlap",
    "📚 Inverted Index"
])

# --- Tab 1: Classify Verse ---
with tab1:
    verse_input = st.text_input("✍️ Input a verse to predict the author:")
    if verse_input:
        processed = preprocess_text(verse_input)
        features = clf_vectorizer.transform([processed])
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][pred]
        label = "Nguyễn Du" if pred == 1 else "Other Author"
        st.success(f"Prediction: **{label}** ({prob:.2%} confidence)")

# --- Tab 2: Cosine Similarity Search ---
with tab2:
    query = st.text_input("🔍 Search by cosine similarity:")
    if query:
        query_vec = search_vectorizer.transform([preprocess_text(query)])
        sims = cosine_similarity(query_vec, search_doc_matrix).flatten()
        top_ids = sims.argsort()[-5:][::-1]
        st.write("📌 Top matching verses:")
        for i in top_ids:
            st.markdown(f"- _{verses[i]}_  \nScore: **{sims[i]:.2f}**")

# --- Tab 3: Overlap Search ---
with tab3:
    overlap_query = st.text_input("🔠 Search by word overlap:")
    if overlap_query:
        query_tokens = set(tokenize(overlap_query))
        scores = [(i, len(query_tokens & set(tokenize(v)))) for i, v in enumerate(verses)]
        top = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
        st.write("📌 Top matching verses:")
        for i, score in top:
            st.markdown(f"- _{verses[i]}_  \nOverlap: **{score}**")

# --- Tab 4: Inverted Index Lookup ---
with tab4:
    keyword = st.text_input("📖 Find all verses containing a word:")
    if keyword:
        word = keyword.strip().lower()
        results = inverted_index.get(word, [])
        if results:
            st.write(f"✅ Found in {len(results)} verse(s):")
            for idx in results[:10]:
                st.markdown(f"- _{verses[idx]}_")
        else:
            st.warning("No matching verses found.")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <p style='text-align: center; font-size: 0.9em;'>
        Made by <b>Melanie</b> with ❤️
    </p>
    """,
    unsafe_allow_html=True
)
