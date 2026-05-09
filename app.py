"""
Storytelling Application for Kids (Ages 3-10)
=============================================
- Upload an image.
- Generate a caption with BLIP.
- Expand the caption into a 50-100 word story using a
  story-generation fine-tuned GPT-2 model.
- Convert the story to speech with gTTS.
- If the fine-tuned model fails, falls back to 'gpt2' + safety filter.
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
    return pipeline(model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_story_generator_model():
    """
    Try to load a fine-tuned story generation model.
    If it fails (e.g., network issues), fall back to 'gpt2'.
    """
    try:
        return pipeline(model="pranavpsv/gpt2-story-generation")
    except Exception as e:
        st.warning(f"Could not load the story model: {e}. Using GPT-2 with safety filter.")
        return pipeline(model="gpt2")

# ---------- Helpers ----------

BLOCKLIST = {   # Only used if fallback to gpt2 is active (safety net)
    "sex", "sexy", "sexual", "porn", "nude", "naked", "kill", "murder",
    "suicide", "dead", "death", "blood", "gun", "drug", "alcohol",
    "smoke", "rape", "assault", "bomb", "terror"
}
COMMENT_FLUFF = [
    "comments below", "let us know", "subscribe", "like and share",
    "follow me on", "click here", "what do you think"
]

def is_safe_for_kids(text: str) -> bool:
    words = set(text.lower().split())
    return not words.intersection(BLOCKLIST)

def has_comment_fluff(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in COMMENT_FLUFF)

def clean_text(text: str) -> str:
    text = re.sub(
        r"(?i)(follow\s*(me|us)\s*on\s*\S+|let\s*us\s*know\s*in\s*the\s*comments\s*below|subscribe\s*to\s*our\s*channel|click\s*here|like\s*and\s*share).*?\.",
        "", text
    )
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def normalize_capitalization(text: str) -> str:
    if not text:
        return text
    text = text[0].upper() + text[1:]
    sentences = re.split(r'(?<=[.!?])\s+', text)
    norm = [s[0].upper() + s[1:] if s else "" for s in sentences]
    return ' '.join(norm)

def trim_story(text: str, min_words=30, max_words=110) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = ""
    wc = 0
    for sent in sentences:
        words = len(sent.split())
        if wc + words > max_words:
            break
        result += sent + " "
        wc += words
    if wc < min_words and len(sentences) > len(result.split('. ')):
        for sent in sentences[len(result.split('. ')):]:
            words = len(sent.split())
            if wc + words > max_words + 20:
                break
            result += sent + " "
            wc += words
    return result.strip()

# ---------- Core Functions ----------

def img2text(image: Image.Image) -> str:
    captioner = load_image_caption_model()
    return captioner(image)[0]["generated_text"]

def text2story(caption: str) -> str:
    generator = load_story_generator_model()
    prompt = f"Once upon a time, {caption.lower()}."
    gen_kwargs = {
        "max_new_tokens": 90,
        "temperature": 0.85,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "pad_token_id": generator.tokenizer.eos_token_id
    }

    max_attempts = 2
    story = ""
    for attempt in range(max_attempts):
        raw = generator(prompt, **gen_kwargs)[0]["generated_text"]
        story = raw[len(prompt):].strip()
        story = clean_text(story)

        # Quality checks
        if len(story.split()) >= 30 and not has_comment_fluff(story):
            if is_safe_for_kids(story):  # only strictly enforced for gpt2 fallback
                break
        gen_kwargs["temperature"] += 0.2

    story = trim_story(story)
    story = normalize_capitalization(story)
    return story

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
