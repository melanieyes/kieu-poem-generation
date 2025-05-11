import streamlit as st
import pickle
import unicodedata
import re
import string
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# --- Page setup ---
st.set_page_config(
    page_title="Truyện Kiều Search & Authorship Prediction",
    page_icon="🌸",
    layout="wide"
)

# --- Inject CSS for layout and image styling ---
st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none; }
        .block-container { padding-top: 1rem; padding-bottom: 2rem; }

        .image-box {
            padding-top: 10px;
            padding-right: 20px;
        }

        .image-box img {
            border-radius: 12px;
            max-width: 100%;
            height: auto;
            object-fit: cover;
        }

        .footer {
            text-align: center;
            font-size: 0.9em;
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- Layout: Left image, right content ---
col1, col2 = st.columns([2, 3], gap="large")

with col1:
    st.markdown(
        '<div class="image-box"><img src="truyenkieu.jpg" width="480"></div>',
        unsafe_allow_html=True
    )

with col2:
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
            processed = unicodedata.normalize('NFC', verse_input.lower()).strip()
            processed = re.sub(r'\s+', ' ', processed)
            features = clf_vectorizer.transform([processed])
            pred = model.predict(features)[0]
            prob = model.predict_proba(features)[0][pred]
            label = "Nguyễn Du" if pred == 1 else "Other Author"
            st.success(f"Prediction: **{label}** ({prob:.2%} confidence)")

    # --- Tab 2: Cosine Similarity Search ---
    with tab2:
        query = st.text_input("🔍 Search by cosine similarity:")
        if query:
            processed = re.sub(r'\s+', ' ', unicodedata.normalize('NFC', query.lower())).strip()
            query_vec = search_vectorizer.transform([processed])
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

    # --- Tab 4: Inverted Index ---
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

# --- Utility functions ---
def tokenize(text):
    text = unicodedata.normalize('NFC', text.lower())
    tokens = text.split()
    return [t.strip(string.punctuation) for t in tokens if t.strip(string.punctuation)]

@st.cache_resource
def load_model():
    try:
        with open("kieu_model.pkl", "rb") as f:
            data = pickle.load(f)
        return data["model"], data["vectorizer"], data["verses"], data["search_vectorizer"], data["search_doc_matrix"]
    except FileNotFoundError:
        st.error("Model file not found.")
        st.stop()

model, clf_vectorizer, verses, search_vectorizer, search_doc_matrix = load_model()

@st.cache_resource
def build_inverted_index(verses):
    index = defaultdict(list)
    for i, verse in enumerate(verses):
        for word in set(tokenize(verse)):
            index[word].append(i)
    return index

inverted_index = build_inverted_index(verses)

# --- Footer ---
st.markdown("<div class='footer'>Made by <b>Melanie</b> with ❤️</div>", unsafe_allow_html=True)
