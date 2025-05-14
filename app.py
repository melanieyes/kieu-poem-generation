import streamlit as st
import unicodedata
import re
import string
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Page Configuration ---
st.set_page_config(
    page_title="Vietnamese Luc Bat Poem Generator",
    page_icon="🌸",
    layout="wide"
)

# --- Inject CSS ---
st.markdown("""
    <style>
        .block-container { padding-top: 0rem !important; }
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
        [data-testid="stSidebar"] { display: none; }
    </style>
""", unsafe_allow_html=True)

# --- Load Image ---
with st.container():
    st.markdown('<div class="centered-container">', unsafe_allow_html=True)
    col_img = st.columns([1, 2, 1])[1]
    with col_img:
        img = Image.open("truyen-kieu.jpg")
        st.image(img, width=600)

# --- Load HuggingFace GPT-2 model ---
@st.cache_resource
def load_poem_model():
    model = AutoModelForCausalLM.from_pretrained("melanieyes/melanie-poem-generation")
    tokenizer = AutoTokenizer.from_pretrained("melanieyes/melanie-poem-generation")
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

gpt_model, gpt_tokenizer, device = load_poem_model()

# --- Generate 3-line Luc-Bat continuation ---
def generate_luc_bat_poem(start_luc, model, tokenizer, device, max_attempts=20):
    def get_tone_class(syllable):
        for c in unicodedata.normalize('NFC', syllable.lower())[::-1]:
            if c in '\u00e0\u00e8\u00ec\u00f2\u00f9\u1ef3': return 'B'
            elif c in '\u00e1\u00e9\u00ed\u00f3\u00fa\u00fd\u1ea3\u1ebb\u1ec9\u1ecf\u1ee7\u1ef7\u00e3\u1ebd\u0129\u00f5\u0169\u1ef9\u1ea1\u1eb9\u1ecb\u1ecd\u1ee5\u1ef5': return 'T'
        return 'B'

    def check_luc(line):
        words = line.split()
        return len(words) == 6 and [get_tone_class(words[i]) for i in [1,3,5]] == ['B','T','B']

    def check_bat(line):
        words = line.split()
        return len(words) == 8 and [get_tone_class(words[i]) for i in [1,3,5,7]] == ['B','T','B','B']

    def generate_next(context, length, checker):
        for _ in range(max_attempts):
            input_ids = tokenizer.encode(context, return_tensors="pt").to(device)
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=30,
                temperature=0.9,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            words = tokenizer.decode(output[0], skip_special_tokens=True).split()
            for i in range(len(words) - length + 1):
                candidate = " ".join(words[i:i+length])
                if checker(candidate): return candidate
        return None

    poem = [start_luc]
    context = start_luc
    pattern = [(8, check_bat), (6, check_luc), (8, check_bat)]

    for length, checker in pattern:
        next_line = generate_next(context, length, checker)
        if not next_line:
            break
        poem.append(next_line)
        context += " " + next_line

    return poem

# --- App Interface ---
st.title("🌟 Vietnamese Lục-Bát Poem Generator")

st.markdown("""
Enter a **Luc** line (6 syllables) to generate the next 3 lines in a traditional **Luc-Bát** poetic form.

> Example input: `mèo con đuổi bóng trăng non`
""")

start_input = st.text_input("🌈 Your starting Luc line (6 syllables):")
if st.button("🖋️ Generate Poem"):
    if not start_input.strip():
        st.error("Please enter a valid 6-syllable Luc line.")
    else:
        poem = generate_luc_bat_poem(start_input, gpt_model, gpt_tokenizer, device)
        if len(poem) >= 4:
            st.success("✅ Generated Luc-Bát Poem:")
            for line in poem:
                st.markdown(f"- _{line}_")
        else:
            st.warning("Only generated part of the poem. Try a different starting line.")

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div class='footer'>
    An AI poetry project inspired by <i>Truyện Kiều</i> and Luc-Bát tradition.<br>
    Model by <b>Melanie</b>, deployed with ❤️ on Hugging Face.
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
