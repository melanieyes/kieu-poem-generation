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
    page_title="Truyện Kiều Search & Authorship Classification",
    page_icon="🌸",
    layout="wide"
)

# --- Load model & data first (to avoid caching errors later) ---
@st.cache_resource
def load_model():
    with open("kieu_model.pkl", "rb") as f:
        data = pickle.load(f)
    return data["model"], data["vectorizer"], data["verses"], data["search_vectorizer"], data["search_doc_matrix"]

model, clf_vectorizer, verses, search_vectorizer, search_doc_matrix = load_model()

# --- Build inverted index ---
@st.cache_resource
def build_inverted_index(verses):
    index = defaultdict(list)
    for i, verse in enumerate(verses):
        for word in set(verse.split()):
            index[word].append(i)
    return index

inverted_index = build_inverted_index(verses)

# --- Utility ---
def tokenize(text):
    text = unicodedata.normalize('NFC', text.lower())
    tokens = text.split()
    return [t.strip(string.punctuation) for t in tokens if t.strip(string.punctuation)]

def preprocess_text(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text.lower())).strip()

# --- Style ---
st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none; }
        html, body, .main {
            height: 100%;
            display: flex;
            flex-direction: column;
        }
        .app-content {
            flex: 1 0 auto;
        }
        .footer {
            flex-shrink: 0;
            text-align: center;
            font-size: 0.9em;
            margin-top: 2rem;
            padding-bottom: 1rem;
        }
        .image-col img {
            border-radius: 12px;
            object-fit: cover;
            width: 100%;
            height: auto;
        }
    </style>
""", unsafe_allow_html=True)

# --- Layout ---
with st.container():
    col1, col2 = st.columns([2, 3], gap="large")

    with col1:
        st.markdown('<div class="image-col">', unsafe_allow_html=True)
        img = Image.open("truyen_kieu.jpg")
        st.image(img, caption=None, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="app-content">', unsafe_allow_html=True)
        st.title("✨ Truyện Kiều Verse Search & Authorship Classifier")

        tab1, tab2, tab3, tab4 = st.tabs([
            "🧠 Classify Verse",
            "📐 Search by Cosine",
            "🔤 Search by Overlap",
            "📚 Inverted Index"
        ])

        with tab1:
            verse_input = st.text_input("✍️ Input a verse to predict the author:")
            if verse_input:
                features = clf_vectorizer.transform([preprocess_text(verse_input)])
                pred = model.predict(features)[0]
                prob = model.predict_proba(features)[0][pred]
                label = "Nguyễn Du" if pred == 1 else "Other Author"
                st.success(f"Prediction: **{label}** ({prob:.2%} confidence)")

        with tab2:
            query = st.text_input("🔍 Search by cosine similarity:")
            if query:
                query_vec = search_vectorizer.transform([preprocess_text(query)])
                sims = cosine_similarity(query_vec, search_doc_matrix).flatten()
                top_ids = sims.argsort()[-5:][::-1]
                st.write("📌 Top matching verses:")
                for i in top_ids:
                    st.markdown(f"- _{verses[i]}_  \nScore: **{sims[i]:.2f}**")

        with tab3:
            overlap_query = st.text_input("🔠 Search by word overlap:")
            if overlap_query:
                query_tokens = set(tokenize(overlap_query))
                scores = [(i, len(query_tokens & set(tokenize(v)))) for i, v in enumerate(verses)]
                top = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
                st.write("📌 Top matching verses:")
                for i, score in top:
                    st.markdown(f"- _{verses[i]}_  \nOverlap: **{score}**")

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
        st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown('<div class="footer">Made by <b>Melanie</b> with ❤️</div>', unsafe_allow_html=True)
