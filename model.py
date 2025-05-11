import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from search import tokenize
import re
from search import preprocess_text

def load_verses(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = []
        for line in f:
            clean_line = re.sub(r'^\d+[.,]*\s*', '', line.strip())
            if clean_line:
                lines.append(preprocess_text(clean_line))
        return lines

# def load_verses(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         return [line.strip() for line in f if line.strip()]


def prepare_data(nguyen_du_verses, other_verses, test_size=0.2):
    X = nguyen_du_verses + other_verses
    y = [1] * len(nguyen_du_verses) + [0] * len(other_verses)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

def extract_features(X_train, X_test, method='tfidf', ngram_range=(1, 2)):
    vectorizer_cls = TfidfVectorizer if method == 'tfidf' else CountVectorizer
    vectorizer = vectorizer_cls(
        tokenizer=tokenize,
        token_pattern=None,
        min_df=2, max_df=0.9,
        ngram_range=ngram_range,
        sublinear_tf=(method == 'tfidf')
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer

def train_classifier(X_train_vec, y_train, classifier_type='svm'):
    if classifier_type == 'svm':
        model = SVC(kernel='linear', probability=True)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_vec, y_train)
    return model

def evaluate_model(model, X_test_vec, y_test):
    y_pred = model.predict(X_test_vec)
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=['Other Author', 'Nguyễn Du']))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Other Author', 'Nguyễn Du'],
                yticklabels=['Other Author', 'Nguyễn Du'])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
    return y_pred

def get_top_features(model, vectorizer, top_n=20):
    if hasattr(model, 'coef_'):
        coefs = model.coef_.toarray()[0]
        top_ids = np.argsort(np.abs(coefs))[-top_n:]
        features = vectorizer.get_feature_names_out()
        return [(features[i], coefs[i]) for i in top_ids]
    return []

def predict_verses(model, vectorizer, verses):
    from search import preprocess_text
    verses_clean = [preprocess_text(v) for v in verses]
    features = vectorizer.transform(verses_clean)
    preds = model.predict(features)
    probs = model.predict_proba(features)
    return [
        {
            "verse": v,
            "prediction": "Nguyễn Du" if preds[i] == 1 else "Other Author",
            "probability": probs[i][1] if preds[i] == 1 else probs[i][0]
        }
        for i, v in enumerate(verses)
    ]

def stylometric_analysis(verses):
    tokens = [token for verse in verses for token in tokenize(verse)]
    return {
        "total_tokens": len(tokens),
        "unique_tokens": len(set(tokens)),
        "avg_verse_length": np.mean([len(tokenize(v)) for v in verses]),
        "top_tokens": Counter(tokens).most_common(20)
    }
