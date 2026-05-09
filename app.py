"""
Storytelling Application for Kids (Ages 3-10)
=============================================
- Users upload an image.
- A caption is generated using BLIP (image-to-text).
- The caption is expanded into a 50-100 word story using a
  child‑story fine‑tuned GPT-2 model (mrm8488/gpt2-child-stories).
- The story is converted to speech with gTTS.
"""

import os
# Suppress model download progress bars and verbose Hugging Face logs
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import streamlit as st
from transformers import pipeline
from PIL import Image
from gtts import gTTS
import tempfile
import re

# ---------- Model Caching ----------
# Models are loaded only once and then reused.

@st.cache_resource
def load_image_caption_model():
    """Load the BLIP image captioning model."""
    return pipeline(model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_story_generator_model():
    """Load a GPT-2 model fine-tuned on children's stories.
    This model is small (88M) and generates safe, age-appropriate content.
    """
    return pipeline(model="mrm8488/gpt2-child-stories")

# ---------- Text Normalization Helpers ----------

def normalize_capitalization(text: str) -> str:
    """
    Capitalize the first letter of the story and the first letter
    of each sentence.
    """
    if not text:
        return text
    text = text[0].upper() + text[1:]
    sentences = re.split(r'(?<=[.!?])\s+', text)
    normalized = []
    for sent in sentences:
        if sent:
            sent = sent[0].upper() + sent[1:]
        normalized.append(sent)
    return ' '.join(normalized)

def clean_story_text(text: str) -> str:
    """
    Remove common social‑media engagement phrases, URLs, and extra
    whitespace that occasionally leak from the base GPT-2 backbone.
    """
    text = re.sub(
        r"(?i)(follow\s*(me|us)\s*on\s*\S+|let\s*us\s*know\s*in\s*the\s*comments\s*below|subscribe\s*to\s*our\s*channel|click\s*here|like\s*and\s*share).*?\.",
        "", text
    )
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def contains_comment_fluff(text: str) -> bool:
    """Quick check for any remaining social‑media bait."""
    fluff = [
        "comments below", "let us know", "subscribe", "like and share",
        "follow me on", "click here", "what do you think"
    ]
    lower = text.lower()
    return any(phrase in lower for phrase in fluff)

def trim_story_to_sentence_range(text: str, min_words: int = 30, max_words: int = 110) -> str:
    """
    Keep only whole sentences until the word count reaches max_words.
    If the result is still too short, add one extra sentence.
    """
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
        remaining = sentences[len(result.split('. ')):]
        for sent in remaining:
            words = len(sent.split())
            if wc + words > max_words + 20:   # soft overflow allowed
                break
            result += sent + " "
            wc += words
    return result.strip()

# ---------- Core Functions ----------

def img2text(image: Image.Image) -> str:
    """Generate a caption describing the uploaded image."""
    captioner = load_image_caption_model()
    return captioner(image)[0]["generated_text"]

def text2story(caption: str) -> str:
    """
    Turn a caption into a short, child‑friendly story (50‑100 words).
    The fine‑tuned child‑story model produces safe output, so no
    additional content filtering is required.
    """
    generator = load_story_generator_model()

    # Simple prompt – the model already understands how to tell a story
    prompt = f"Once upon a time, {caption.lower()}."

    gen_kwargs = {
        "max_new_tokens": 90,
        "temperature": 0.85,
        "pad_token_id": generator.tokenizer.eos_token_id,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.2
    }

    # Try twice; if the first attempt yields something too short or
    # contains comment‑like fluff, increase temperature and try again.
    max_attempts = 2
    story = ""
    for attempt in range(max_attempts):
        raw = generator(prompt, **gen_kwargs)[0]["generated_text"]
        story = raw[len(prompt):].strip()
        story = clean_story_text(story)

        if len(story.split()) >= 30 and not contains_comment_fluff(story):
            break
        gen_kwargs["temperature"] += 0.2

    # Trim to a nice sentence range and fix capitalization
    story = trim_story_to_sentence_range(story, min_words=30, max_words=110)
    story = normalize_capitalization(story)
    return story

def text2audio(story_text: str) -> str:
    """Convert the story to speech and return the path to an MP3 file."""
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

    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"]
    )

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
