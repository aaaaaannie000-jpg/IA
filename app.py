"""
Storytelling Application for Kids (Ages 3-10)
=============================================
- Users upload an image.
- The app generates a caption using an image-to-text model.
- The caption is expanded into a 50-100 word story for children.
- The story is converted to speech and can be played directly.
"""

import os
# Hide model download progress bars and reduce console noise
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import streamlit as st
from transformers import pipeline
from PIL import Image
from gtts import gTTS
import tempfile
import re

# ---------- Model Loading with Streamlit Caching ----------

@st.cache_resource
def load_image_caption_model():
    """Load the image captioning model."""
    return pipeline(model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_story_generator_model():
    """Load GPT-2 (same model used in class demo). 
    If deployment size is an issue, fall back to 'distilgpt2'.
    """
    return pipeline(model="gpt2")      # Same as the teacher's demo

# ---------- Text Normalization Helpers ----------

def normalize_capitalization(text: str) -> str:
    """Make the story start with a capital letter and capitalize 
    the first letter of each sentence."""
    if not text:
        return text
    # Capitalize the first character of the whole text
    text = text[0].upper() + text[1:]
    # Capitalize the first letter after every sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    normalized = []
    for sent in sentences:
        if sent:
            sent = sent[0].upper() + sent[1:]
        normalized.append(sent)
    return ' '.join(normalized)

def clean_story_text(text: str) -> str:
    """Remove common social media / comment fluff from generated text."""
    text = re.sub(
        r"(?i)(follow\s*(me|us)\s*on\s*\S+|let\s*us\s*know\s*in\s*the\s*comments\s*below|subscribe\s*to\s*our\s*channel|click\s*here|like\s*and\s*share).*?\.", 
        "", text
    )
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def contains_comment_fluff(text: str) -> bool:
    """Return True if the text still contains typical engagement-bait phrases."""
    fluff = [
        "comments below", "let us know", "subscribe", "like and share",
        "follow me on", "click here", "what do you think"
    ]
    lower = text.lower()
    return any(phrase in lower for phrase in fluff)

# ---------- Core Functions ----------

def img2text(image: Image.Image) -> str:
    captioner = load_image_caption_model()
    return captioner(image)[0]["generated_text"]

def text2story(caption: str) -> str:
    """
    Generate a 50-100 word kid-friendly story from a caption.
    Uses GPT-2 in a continuation style, similar to the classroom example.
    Falls back to a second attempt if the first output is poor.
    """
    generator = load_story_generator_model()
    
    # Classroom-style prompt: just give the story beginning
    prompt = f"Once upon a time, there was {caption.lower()}. "
    
    gen_kwargs = {
        "max_new_tokens": 120,
        "temperature": 0.9,
        "pad_token_id": generator.tokenizer.eos_token_id,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.2
    }
    
    max_attempts = 2
    story = ""
    for attempt in range(max_attempts):
        raw = generator(prompt, **gen_kwargs)[0]["generated_text"]
        story = raw[len(prompt):].strip()
        story = clean_story_text(story)
        # Quality check: at least 30 words and no comment fluff
        if len(story.split()) >= 30 and not contains_comment_fluff(story):
            break
        # If not good enough, increase temperature for more variety next try
        gen_kwargs["temperature"] += 0.2
    
    # Trim to a reasonable number of complete sentences
    story = trim_story_to_sentence_range(story, min_words=30, max_words=110)
    # Fix capitalization
    story = normalize_capitalization(story)
    return story

def trim_story_to_sentence_range(text: str, min_words: int = 30, max_words: int = 110) -> str:
    """Keep only whole sentences until max_words is reached."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = ""
    wc = 0
    for sent in sentences:
        words = len(sent.split())
        if wc + words > max_words:
            break
        result += sent + " "
        wc += words
    # If too short, try adding one more sentence even if it exceeds slightly
    if wc < min_words and len(sentences) > len(result.split('. ')):
        for sent in sentences[len(result.split('. ')):]:
            words = len(sent.split())
            if wc + words > max_words + 20:
                break
            result += sent + " "
            wc += words
    return result.strip()

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
