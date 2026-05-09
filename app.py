"""
Storytelling Application for Kids (Ages 3-10)
=============================================
- Users upload an image.
- The app generates a caption using an image-to-text model.
- The caption is expanded into a 50-100 word story for children.
- The story is converted to speech and can be played directly.
"""

import os
# Suppress Hugging Face download progress bars and verbose logs
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import streamlit as st
from transformers import pipeline
from PIL import Image
from gtts import gTTS
import tempfile
import re

# ---------- Model Loading with Streamlit Caching ----------
# Models are cached to avoid re-downloading on every interaction.

@st.cache_resource
def load_image_caption_model():
    """Load the image captioning model."""
    return pipeline(model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_story_generator_model():
    """Load the text generation model.
    Uses 'distilgpt2' for smaller footprint; upgrade to 'gpt2' if resources allow.
    """
    return pipeline(model="distilgpt2")

# ---------- Core Functions ----------

def img2text(image: Image.Image) -> str:
    """
    Generate a descriptive caption from an uploaded image.

    Args:
        image (PIL.Image): The input image in RGB format.

    Returns:
        str: The generated caption.
    """
    captioner = load_image_caption_model()
    return captioner(image)[0]["generated_text"]

def text2story(caption: str) -> str:
    """
    Expand a caption into a short, fun story for children (50-100 words).

    Uses a carefully crafted prompt and post-processing to improve story quality.

    Args:
        caption (str): The image caption.

    Returns:
        str: A cleaned, child-friendly story.
    """
    generator = load_story_generator_model()
    
    # Build a kid-friendly prompt that guides the model to tell a story
    prompt = f"Once upon a time, there was {caption.lower()}.\nA little story for kids:"
    
    result = generator(
        prompt,
        max_new_tokens=100,
        temperature=0.9,
        pad_token_id=generator.tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.15
    )[0]["generated_text"]
    
    # Extract only the generated part (remove the prompt)
    story = result[len(prompt):].strip()
    
    # Post-processing: trim to complete sentences and keep roughly 50-100 words
    sentences = re.split(r'(?<=[.!?])\s+', story)
    clean_story = ""
    word_count = 0
    
    for sent in sentences:
        words = len(sent.split())
        if word_count + words > 110:   # soft cap to avoid abrupt cut-off
            break
        clean_story += sent + " "
        word_count += words
    
    return clean_story.strip()

def text2audio(story_text: str) -> str:
    """
    Convert story text to speech using gTTS and save as a temporary MP3 file.

    Args:
        story_text (str): The story.

    Returns:
        str: File path to the temporary audio file.
    """
    tts = gTTS(story_text, lang="en")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name

# ---------- Streamlit User Interface ----------

def main():
    st.set_page_config(page_title="Magic Storyteller", page_icon="📖")
    st.title("📸 Magic Storyteller for Kids")
    st.markdown("**A fun way to turn any picture into a story!** 🌟")
    st.write("Upload an image, and I'll tell you a magical story!")

    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Your Picture", use_container_width=True)

        # Processing steps
        with st.spinner("✨ Creating your story... This may take a few seconds."):
            # 1. Image → caption
            caption = img2text(image)
            st.subheader("📝 What I See")
            st.write(caption)

            # 2. Caption → story
            story = text2story(caption)
            st.subheader("📚 Your Story")
            st.write(story)

            # 3. Story → audio
            audio_file = text2audio(story)
            st.subheader("🔊 Listen to Your Story")
            st.audio(audio_file, format="audio/mp3")

if __name__ == "__main__":
    main()
