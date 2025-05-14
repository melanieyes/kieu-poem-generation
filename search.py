import re
import string
import unicodedata
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_text(text):
    text = unicodedata.normalize('NFC', text.lower())
    text = re.sub(r'^\d+[.:,]*\s*', '', text)
    text = re.sub(r'[{}]'.format(re.escape(string.punctuation)), '', text)
    return re.sub(r'\s+', ' ', text).strip()

def tokenize(text):
    text = preprocess_text(text)
    return [t for t in text.split() if t]

def search_by_cosine(query, verses, vectorizer, doc_matrix, top_k=5):
    query_vec = vectorizer.transform([preprocess_text(query)])
    sims = cosine_similarity(query_vec, doc_matrix).flatten()
    top_ids = sims.argsort()[-top_k:][::-1]
    return [(verses[i], round(sims[i], 4)) for i in top_ids]

def search_by_overlap(query, verses, top_k=5):
    query_tokens = set(tokenize(query))
    scores = [(i, len(query_tokens & set(tokenize(v)))) for i, v in enumerate(verses)]
    top = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    return [(verses[i], score) for i, score in top]

def build_inverted_index(verses):
    index = defaultdict(list)
    for i, verse in enumerate(verses):
        for word in set(tokenize(verse)):
            index[word].append(i)
    return index