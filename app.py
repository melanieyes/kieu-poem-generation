import streamlit as st
import pickle
import unicodedata
import re
import string
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# --- Page configuration ---
st.set_page_config(
    page_title="Truyện Kiều Search & Authorship Prediction",
    page_icon="🌸",
    layout="wide"
)

# --- Inject CSS to center content and style footer ---
st.markdown("""
    <style>
        .centered-container {
            max-width: 900px;
            margin: 0 auto;
            padding: 1rem 2rem;
        }
        .footer {
            text-align: center;
            font-size: 0.9em;
            margin-top: 3rem;
            color: #999;
        }
        [data-testid="stSidebar"] {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# --- Preprocessing functions ---
def tokenize(text):
    text = unicodedata.normalize('NFC', text.lower())
    tokens = text.split()
    return [t.strip(string.punctuation) for t in tokens if t.strip(string.punctuation)]

def preprocess_text(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text.lower())).strip()

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

# --- Main Centered UI Layout ---
with st.container():
    st.markdown('<div class="centered-container">', unsafe_allow_html=True)

    # Banner Image
    img = Image.open("truyen-kieu.jpg")
    st.image(img, width=600)

    # Title
    st.title("✨ Truyện Kiều Search & Authorship Classifier")

    # --- Tabs UI ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧠 Classify Verse",
        "📐 Search by Cosine",
        "🔤 Search by Overlap",
        "📚 Inverted Index"
    ])

    # --- Tab 1: Author Classifier ---
    with tab1:
        verse_input = st.text_input("✍️ Input a verse to predict the author:")
        if verse_input:
            features = clf_vectorizer.transform([preprocess_text(verse_input)])
            pred = model.predict(features)[0]
            prob = model.predict_proba(features)[0][pred]
            label = "Nguyễn Du" if pred == 1 else "Other Author"
