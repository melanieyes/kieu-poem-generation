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
  - Fine-tuned Vietnamese GPT-2 (`danghuy1999/gpt2-viwiki`)
- Includes tone-checking and formatting to preserve **lục bát** poetic structure.

### 3. **Multimodal AI (Gemini Experiments)**
- Generate images from *Truyện Kiều* verses using Gemini (text-to-image).
- Retrieve the most semantically relevant verse for a given image (image-to-text).
- Documented in the final project report.

---

## 🌐 Live Demo

You can try the Streamlit app here:  
👉 **[https://master-77ujvbqhvxmw2yrndstdjx.streamlit.app/](https://master-77ujvbqhvxmw2yrndstdjx.streamlit.app/)**

---

## 🧪 How to Run Locally

### Search & Classifier:
```bash
python app.py  # Launches Streamlit interface for verse search and authorship classification
