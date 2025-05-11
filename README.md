
---

## 🔍 Features

### 1. **Search & Authorship Attribution**
- Built using `TF-IDF` vectorization.
- `search.py`: Implements keyword search and cosine similarity search on Truyện Kiều.
- `model.py`: Contains SVM classifier for authorship attribution (Nguyễn Du vs. others).
- `haku.csv`: Contrastive corpus for non-Nguyễn Du verses.
- Achieved ~91% accuracy in authorship prediction.

### 2. **Verse Generation (Language Modeling)**
- Implemented in `task2_kieu_generation.ipynb`.
- Two approaches:
  - From-scratch Transformer model (PyTorch)
  - Fine-tuning GPT-2 (`danghuy1999/gpt2-viwiki`)
- Includes tone-checking and formatting to preserve **lục bát** poetic structure.

### 3. **Multimodal AI (Gemini Experiments)**
- Generate images from *Truyện Kiều* verses using Gemini.
- Retrieve the most semantically relevant verse for a given image.
- Documented in final report (see PDF).

---

## 🧪 How to Run

### Search & Classifier:
```bash
python app.py  # Entry point for search interface and classification
