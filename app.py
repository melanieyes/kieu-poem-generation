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

# --- Inject CSS to center, remove top padding, and style ---
st.markdown("""
    <style>
        .block-container {
            padding-top: 0rem !important;
        }
        .centered-container {
            max-width: 900px;
            margin: 0 auto;
            padding: 0rem 2rem 1rem 2rem;
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

# --- Layout ---
with st.container():
    st.markdown('<div class="centered-container">', unsafe_allow_html=True)

    # Centered Image
    col_img = st.columns([1, 2, 1])[1]
    with col_img:
        img = Image.open("truyen-kieu.jpg")
        st.image(img, width=600)

    # Title
    st.title("✨ Truyện Kiều Verse Search & Authorship Classifier")

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧠 Classify Verse",
        "📐 Search by Cosine",
        "🔤 Search by Overlap",
        "📚 Inverted Index"
    ])

    # --- Tab 1: Classify ---
    with tab1:
        verse_input = st.text_input("✍️ Input a verse to predict the author:")
        if verse_input:
            features = clf_vectorizer.transform([preprocess_text(verse_input)])
            pred = model.predict(features)[0]
            prob = model.predict_proba(features)[0][pred]
            label = "Nguyễn Du" if pred == 1 else "Other Author"
            st.success(f"Prediction: **{label}** ({prob:.2%} confidence)")

    # --- Tab 2: Cosine Similarity ---
    with tab2:
        query = st.text_input("🔍 Search by cosine similarity:")
        if query:
            query_vec = search_vectorizer.transform([preprocess_text(query)])
            sims = cosine_similarity(query_vec, search_doc_matrix).flatten()
            top_ids = sims.argsort()[-5:][::-1]
            st.write("📌 Top matching verses:")
            for i in top_ids:
                st.markdown(f"- _{verses[i]}_  \nScore: **{sims[i]:.2f}**")

    # --- Tab 3: Overlap ---
    with tab3:
        overlap_query = st.text_input("🔠 Search by word overlap:")
        if overlap_query:
            query_tokens = set(tokenize(overlap_query))
            scores = [(i, len(query_tokens & set(tokenize(v)))) for i, v in enumerate(verses)]
            top = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
            st.write("📌 Top matching verses:")
            for i, score in top:
                st.markdown(f"- _{verses[i]}_  \nOverlap: **{score}**")

    # --- Tab 4: Inverted Index ---
    with tab4:
        keyword = st.text_input("📖 Find all verses containing a word:")
        if keyword:
            word = keyword.strip().lower()
            results = inverted_index.get(word, [])
            if results:
                st.write(f"✅ Found in {len(results)} verse(s):")
                for idx in results[:15]:
                    st.markdown(f"- _{verses[idx]}_")
            else:
                st.warning("No matching verses found.")

   # --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div class='footer'>
        An AI project exploring the beauty of Nguyễn Du’s <i>Truyện Kiều</i><br>
        Made by <b>Melanie</b> with ❤️
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)
