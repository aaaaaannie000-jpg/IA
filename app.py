"""
Storytelling Application for Kids (Ages 3-10)
=============================================
- Upload an image.
- Generate a caption using BLIP image-to-text.
- Expand the caption into a 50-100 word children's story with GPT-2.
- Convert the story to speech with gTTS.
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
    """Load BLIP image captioning model."""
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_story_generator_model():
    """Load GPT-2 for story generation (same as teacher's demo)."""
    return pipeline("text-generation", model="gpt2")

# ---------- Simple text helpers ----------

def clean_story_text(text:str) -> str:
    """Remove occasional social media/comment fluff."""
    # Remove phrases like "This is what we have seen from this photo!", "It's amazing how awesome it looks..."
    text = re.sub(r"(?i)(this is what we have seen|it['’]?s amazing how awesome it looks|subscribe|follow me|check out|click here|let us know in the comments).*?[.!?]", "", text)
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def trim_to_sentence_range(text:str, min_words=30, max_words=110) -> str:
    """Keep only complete sentences up to ~100 words."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = ""
    word_count = 0
    for sent in sentences:
        words = len(sent.split())
        if word_count + words > max_words:
            break
        result += sent + " "
        word_count += words
    return result.strip()

# ---------- Core Functions ----------

def img2text(image:Image.Image) -> str:
    """Generate a descriptive caption from the uploaded image."""
    captioner = load_image_caption_model()
    return captioner(image)[0]["generated_text"]

def text2story(caption:str) -> str:
    """
    Turn the caption into a 50-100 word children's story.
    Uses a simple fairy-tale opening that GPT-2 will naturally continue.
    """
    generator = load_story_generator_model()

    # A clean story opening — the model will continue from here.
    prompt = f"Once upon a time, there was {caption}."

    raw = generator(
        prompt,
        max_new_tokens=150,          # enough for ~100 words
        temperature=0.85,
        pad_token_id=generator.tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2
    )[0]["generated_text"]

    # Extract the whole story (opening + continuation)
    story = raw.strip()

    # Light cleanup: remove occasional social-media phrases
    story = clean_story_text(story)

    # Trim to 50-100 words, keeping whole sentences
    story = trim_to_sentence_range(story, min_words=50, max_words=100)

    return story

def text2audio(story_text:str) -> str:
    """Convert story text to speech and return path to MP3."""
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

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

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
