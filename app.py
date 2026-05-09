"""
Storytelling Application for Kids (Ages 3-10)
=============================================
- Upload an image
- Generate a caption with BLIP (image-to-text)
- Generate two story versions with distilgpt2
- Use sentiment analysis to select the more positive story
- Convert the story to speech with gTTS
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
    """Load distilgpt2 for story generation."""
    return pipeline("text-generation", model="distilgpt2")

@st.cache_resource
def load_sentiment_model():
    """Load sentiment analysis model (same as class example)."""
    return pipeline("sentiment-analysis",
                    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# ---------- Helper to trim story to 50-100 words ----------

def trim_to_sentence_range(text: str, min_words=30, max_words=110) -> str:
    """Keep complete sentences up to ~100 words."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = ""
    wc = 0
    for sent in sentences:
        words = len(sent.split())
        if wc + words > max_words:
            break
        result += sent + " "
        wc += words
    return result.strip()

# ---------- Core Functions ----------

def img2text(image: Image.Image) -> str:
    captioner = load_image_caption_model()
    return captioner(image)[0]["generated_text"]

def text2story(caption: str) -> str:
    """
    Generate two story continuations, then use sentiment analysis
    to pick the more positive one (teacher's method).
    """
    generator = load_story_generator_model()
    sentiment = load_sentiment_model()

    # Simple opening – the model will continue from here
    prompt = f"Once upon a time, {caption}."

    # Generate two story candidates
    story1 = generator(prompt, max_new_tokens=150, temperature=0.85,
                       pad_token_id=generator.tokenizer.eos_token_id,
                       do_sample=True, top_k=50, top_p=0.9,
                       repetition_penalty=1.2,
                       num_return_sequences=1)[0]["generated_text"]
    story2 = generator(prompt, max_new_tokens=150, temperature=0.85,
                       pad_token_id=generator.tokenizer.eos_token_id,
                       do_sample=True, top_k=50, top_p=0.9,
                       repetition_penalty=1.2,
                       num_return_sequences=1)[0]["generated_text"]

    # Sentiment analysis
    result1 = sentiment(story1)[0]
    result2 = sentiment(story2)[0]

    # Choose the more positive story (teacher's logic)
    if result1["label"] == "POSITIVE" and result2["label"] != "POSITIVE":
        chosen = story1
    elif result2["label"] == "POSITIVE" and result1["label"] != "POSITIVE":
        chosen = story2
    elif result1["label"] == "POSITIVE" and result2["label"] == "POSITIVE":
        if result1["score"] >= result2["score"]:
            chosen = story1
        else:
            chosen = story2
    else:
        # If both negative, just pick the first one (fallback)
        chosen = story1

    # Trim to complete sentences within ~100 words
    return trim_to_sentence_range(chosen, min_words=30, max_words=110)

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
