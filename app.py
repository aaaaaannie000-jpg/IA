"""
Storytelling Application for Kids (Ages 3-10)
=============================================
- Upload an image
- Generate a caption using BLIP image-to-text
- Expand the caption into a 50-100 word children's story with GPT-2
- Automatically select the best story among multiple candidates
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
    """Load GPT-2 for story generation."""
    return pipeline("text-generation", model="gpt2")

# ---------- Text helpers ----------

def is_comment_fluff(text: str) -> bool:
    """Check if the text contains typical comment/youtube fluff."""
    fluff_patterns = [
        r"subscribe",
        r"like and share",
        r"follow me",
        r"click here",
        r"let us know in the comments",
        r"that['’]?s right",
        r"check out",
        r"what do you think",
        r"i made my first video",
        r"re[- ]released on",
        r"my favourite rides",
    ]
    text_lower = text.lower()
    for pat in fluff_patterns:
        if re.search(pat, text_lower):
            return True
    return False

def story_length_ok(text: str, min_words=40, max_words=120) -> bool:
    words = text.split()
    return min_words <= len(words) <= max_words

def trim_to_sentence_end(text: str, max_words=110) -> str:
    """Trim text to complete sentences within max_words."""
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

def score_story(text: str) -> int:
    """
    Score how good a story is. Higher is better.
    2 points for no fluff, 1 point for correct length, 1 point for ending with punctuation.
    """
    score = 0
    if not is_comment_fluff(text):
        score += 2
    if story_length_ok(text, 40, 120):
        score += 1
    if text and text[-1] in ".!?":
        score += 1
    return score

# ---------- Core Functions ----------

def img2text(image:Image.Image) -> str:
    captioner = load_image_caption_model()
    return captioner(image)[0]["generated_text"]

def text2story(caption:str) -> str:
    """
    Generate 3 candidate stories and return the best one.
    If none are good, retry with higher temperature.
    """
    generator = load_story_generator_model()
    prompt = f"Once upon a time, there was {caption}."
    
    best_story = ""
    best_score = -1
    attempts = 2
    temp = 0.85

    for attempt in range(attempts):
        # Generate 3 candidates
        results = generator(
            prompt,
            max_new_tokens=150,
            temperature=temp,
            pad_token_id=generator.tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            num_return_sequences=3
        )
        
        for r in results:
            full_text = r["generated_text"].strip()
            # Clean possible fluff (just in case)
            # Actually we rely on scoring, but we can do a light clean
            story_candidate = full_text
            if len(story_candidate) < len(prompt):
                continue
            # Trim to roughly 100 words
            trimmed = trim_to_sentence_end(story_candidate, 100)
            s = score_story(trimmed)
            if s > best_score:
                best_score = s
                best_story = trimmed
        
        # If we already have a good story (score >= 4), break
        if best_score >= 4:
            break
        # Otherwise, increase temperature to get more variety
        temp += 0.15

    # If still no good story, just return the best we have (or the first one)
    if not best_story:
        best_story = "Once upon a time, there was " + caption + ". They lived happily ever after. The end."
    
    return best_story

def text2audio(story_text:str) -> str:
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
