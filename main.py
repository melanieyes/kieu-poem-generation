from model import (
    load_verses, prepare_data, extract_features,
    train_classifier, evaluate_model, get_top_features,
    predict_verses, stylometric_analysis
)
from search import (
    search_by_cosine, search_by_overlap,
    build_inverted_index, preprocess_text, tokenize
)
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# ----------- Load and Preprocess Data ----------
nd_verses = load_verses("truyen_kieu_data.txt")
other_verses = load_verses("haku.csv")

# ----------- Vectorize Nguyễn Du's verses for Search ----------
vectorizer = TfidfVectorizer(
    tokenizer=tokenize,
    token_pattern=None,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9
)
doc_matrix = vectorizer.fit_transform(nd_verses)

# ----------- Run Search Engine Queries ----------
print("\n=== Cosine Similarity Search ===")
print(search_by_cosine("vầng trăng", nd_verses, vectorizer, doc_matrix))

print("\n=== Token Overlap Search ===")
print(search_by_overlap("vầng trăng", nd_verses))

print("\n=== Inverted Index Example ===")
inverted = build_inverted_index(nd_verses)
print(f"Lines with 'trăng': {inverted['trăng']}")

# ----------- Train Classifier ----------
X_train, X_test, y_train, y_test = prepare_data(nd_verses, other_verses)
X_train_vec, X_test_vec, clf_vectorizer = extract_features(X_train, X_test)
model = train_classifier(X_train_vec, y_train)

# ----------- Evaluate ----------
evaluate_model(model, X_test_vec, y_test)

# ----------- Save for Streamlit ----------
search_vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2), min_df=2, max_df=0.9)
search_doc_matrix = search_vectorizer.fit_transform(nd_verses)

with open("kieu_model.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "vectorizer": clf_vectorizer,
        "verses": nd_verses,
        "search_vectorizer": search_vectorizer,
        "search_doc_matrix": search_doc_matrix
    }, f)


# ----------- Show Top Features ----------
print("\n=== Top Features ===")
for feat, score in get_top_features(model, clf_vectorizer):
    print(f"{feat:15s}: {score:.4f}")

# ----------- Sample Predictions ----------
print("\n=== Prediction Example ===")
print(predict_verses(model, clf_vectorizer, [
    'Thương sao cho trọn thì thương',
    'Ve kêu không mỏi mệt'
]))

# ----------- Stylometry ----------
print("\n=== Stylometric Analysis ===")
print(stylometric_analysis(nd_verses))
