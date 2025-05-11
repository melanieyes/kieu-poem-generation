# Truyện Kiều AI Project

This project applies Artificial Intelligence techniques to Vietnamese classical literature, focusing on Nguyễn Du's *Truyện Kiều*. It combines natural language processing, machine learning, and multimodal exploration to analyze, classify, and interact with poetic verses.

---

## 🔗 Live Demo

👉 [Try the Streamlit App](https://truyen-kieu-poem-generation-melanie.streamlit.app/)  
Interactively search verses, classify authorship, and explore Vietnamese poetic structure.

---

## 🔍 Features

### 1. **Search & Authorship Attribution**
- Built using `TF-IDF` vectorization.
- `search.py`: Implements keyword search, cosine similarity search, and inverted index lookup.
- `model.py`: Contains SVM classifier for authorship attribution (Nguyễn Du vs. others).
- `haku.csv`: Contrastive corpus for non-Nguyễn Du verses.
- Achieved ~91% accuracy in authorship prediction.

### 2. **Verse Generation (Language Modeling)**
- Implemented in `task2_kieu_generation.ipynb`.
- Two approaches:
  - From-scratch Transformer model (PyTorch)
  - Fine-tuned GPT-2 (`danghuy1999/gpt2-viwiki`)
- Includes tone-checking and formatting to preserve **lục bát** poetic structure.

### 3. **Multimodal AI (Gemini Experiments)**
- Generate images from *Truyện Kiều* verses using Gemini (text-to-image).
- Retrieve the most semantically relevant verse for a given image (image-to-text).
- Documented in the final project report.

---

## 🧪 How to Run Locally

### 1. Launch the App:
```bash
streamlit run app.py
