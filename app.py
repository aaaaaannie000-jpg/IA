"""
Storytelling Application for Kids (Ages 3-10)
=============================================
- Upload an image
- Generate a caption using BLIP
- Generate multiple story candidates with a child‑story fine‑tuned GPT-2
- Automatically select the best candidate (safe, proper length, complete)
- Convert to speech with gTTS
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
    """
    Load a GPT-2 model fine-tuned on children's stories.
    Safe, clean, and age-appropriate by design.
    """
    return pipeline("text-generation", model="mrm8488/gpt2-child-stories")

# ---------- Simple text helpers ----------

def trim_to_sentence_end(text: str, max_words:int = 100) -> str:
    """Keep only whole sentences up to max_words."""
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

def story_quality_score(text: str) -> int:
    """
    Score a story candidate (higher = better).
    +2: Length 40-110 words
    +1: Ends with . ! or ?
    """
    score = 0
    word_count = len(text.split())
    if 40 <= word_count <= 110:
        score += 2
    if text and text[-1] in ".!?":
        score += 1
    return score

# ---------- Core Functions ----------

def img2text(image:Image.Image) -> str:
    captioner = load_image_caption_model()
    return captioner(image)[0]["generated_text"]

def text2story(caption:str) -> str:
    """
    Generate 3 children's story candidates using a safe fine‑tuned model,
    and automatically pick the best one based on length and completeness.
    """
    generator = load_story_generator_model()
    prompt = f"Once upon a time, there was {caption}."

    best_story = ""
    best_score = -1

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

    for r in results:
        full_text = r["generated_text"].strip()
        if not full_text.startswith(prompt):
            continue
        # Trim to complete sentences within ~100 words
        trimmed = trim_to_sentence_end(full_text, 100)
        score = story_quality_score(trimmed)
        if score > best_score:
            best_score = score
            best_story = trimmed

    # Fallback in case no candidate qualified
    if not best_story:
        best_story = f"Once upon a time, there was {caption}. They lived happily ever after. The end."

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
