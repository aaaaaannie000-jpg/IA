"""
Storytelling Application for Kids (Ages 3-10)
=============================================
- Upload an image
- Generate a caption using BLIP (Salesforce/blip-image-captioning-base)
- Expand the caption into a 50–100 word children's story with distilgpt2
- Uses teacher's continuation method + multi-candidate selection
- Converts the story to speech with gTTS
"""

import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import streamlit as st
from transformers import pipeline
from PIL import Image
from gtts import gTTS
import tempfile
import re

# ---------- Model Caching ----------

@st.cache_resource
def load_image_caption_model():
    """Load BLIP image captioning model (as recommended by the assignment)."""
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_story_generator_model():
    """Load distilgpt2 – lightweight and exactly what the skeleton specified."""
    return pipeline("text-generation", model="distilgpt2")

# ---------- Simple text helpers ----------

def is_clean_sentence(text: str) -> bool:
    """
    Return False if the text looks like leftover social media fluff,
    article clippings, or random punctuation fragments.
    """
    lower = text.lower()
    # Common garbage patterns from distilgpt2
    garbage_patterns = [
        r"stuff$",                     # lines that just end with "Stuff"
        r"it's just that",            # typical blog/comment phrasing
        r"that's right",
        r"check out",
        r"subscribe",
        r"follow me",
        r"let us know",
        r"what do you think",
        r"i made my first",
        r"re[- ]released",
        r"click here",
    ]
    for pat in garbage_patterns:
        if re.search(pat, lower):
            return False
    # Also discard very short sentences (< 3 words) that are likely noise
    if len(text.split()) < 3:
        return False
    return True

def trim_story(text: str, max_words: int = 110) -> str:
    """
    Keep only complete, clean sentences up to ~100 words.
    """
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    clean_sentences = [s.strip() for s in raw_sentences if is_clean_sentence(s.strip())]
    
    result = ""
    wc = 0
    for sent in clean_sentences:
        words = len(sent.split())
        if wc + words > max_words:
            break
        result += sent + " "
        wc += words
    return result.strip()

def story_candidate_score(text: str) -> int:
    """
    Score a candidate story. Higher = better.
    - 40‑110 words: +2
    - Ends with . ! or ?: +1
    """
    score = 0
    word_count = len(text.split())
    if 40 <= word_count <= 110:
        score += 2
    if text and text[-1] in ".!?":
        score += 1
    return score

# ---------- Core Functions ----------

def img2text(image: Image.Image) -> str:
    captioner = load_image_caption_model()
    return captioner(image)[0]["generated_text"]

def text2story(caption: str) -> str:
    """
    Generate 3 story continuations from distilgpt2 using the teacher's
    minimal-prompt method, then pick the best one.
    """
    generator = load_story_generator_model()
    prompt = f"Once upon a time, there was {caption}."

    # Generate 3 candidates
    results = generator(
        prompt,
        max_new_tokens=150,
        temperature=0.85,
        pad_token_id=generator.tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        num_return_sequences=3
    )

    best_story = ""
    best_score = -999

    for r in results:
        full_text = r["generated_text"].strip()
        # Trim to clean, complete sentences
        trimmed = trim_story(full_text, 100)
        if not trimmed:
            continue
        s = story_candidate_score(trimmed)
        if s > best_score:
            best_score = s
            best_story = trimmed

    if not best_story:
        best_story = f"Once upon a time, there was {caption}. They lived happily ever after. The end."

    return best_story

def text2audio(story_text: str) -> str:
    tts = gTTS(story_text, lang="en")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name

# ---------- Streamlit UI ----------

def main():
    st.set_page_config(page_title="Magic Storyteller", page_icon="📖")
    st.title("📸 Magic Storyteller for Kids")
    st.markdown("**A fun way to turn any picture into a story!** 🌟")
    st.write("Upload an image, and I'll tell you a magical story!")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Your Picture", use_container_width=True)

        with st.spinner("✨ Creating your story... This may take a few seconds."):
            caption = img2text(image)
            st.subheader("📝 What I See")
            st.write(caption)

            story = text2story(caption)
            st.subheader("📚 Your Story")
            st.write(story)

            audio_file = text2audio(story)
            st.subheader("🔊 Listen to Your Story")
            st.audio(audio_file, format="audio/mp3")

if __name__ == "__main__":
    main()
