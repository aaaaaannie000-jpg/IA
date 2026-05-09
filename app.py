"""
Storytelling Application for Kids (Ages 3-10)
=============================================
- Users upload an image.
- The app generates a caption using an image-to-text model.
- The caption is expanded into a 50-100 word story for children.
- The story is converted to speech and can be played directly.
"""

import streamlit as st
from transformers import pipeline
from PIL import Image
from gtts import gTTS
import tempfile

# ---------- Model Loading with Streamlit Caching ----------
# This ensures models are downloaded only once and reused.

@st.cache_resource
def load_image_caption_model():
    """Load the image captioning model."""
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_story_generator_model():
    """Load the text generation model (lightweight for deployment)."""
    return pipeline("text-generation", model="distilgpt2")

# ---------- Core Functions ----------

def img2text(image: Image.Image) -> str:
    """
    Generate a descriptive caption from an uploaded image.

    Args:
        image (PIL.Image): The input image.

    Returns:
        str: The generated caption.
    """
    captioner = load_image_caption_model()
    caption = captioner(image)[0]["generated_text"]
    return caption

def text2story(caption: str) -> str:
    """
    Expand a caption into a short, fun story for children (50-100 words).

    Args:
        caption (str): The image caption.

    Returns:
        str: The generated story.
    """
    generator = load_story_generator_model()
    # Craft a prompt that guides the model to write a child-friendly story
    prompt = (
        f"Write a short, fun story for kids (50-100 words) based on this scene: {caption}\n"
        "Story:"
    )
    result = generator(
        prompt,
        max_new_tokens=120,
        temperature=0.85,
        pad_token_id=generator.tokenizer.eos_token_id,
        do_sample=True
    )[0]["generated_text"]
    # Extract only the generated story (remove the prompt)
    story = result[len(prompt):].strip()
    return story

def text2audio(story_text: str) -> str:
    """
    Convert story text to speech and save as an MP3 file.

    Args:
        story_text (str): The story.

    Returns:
        str: Path to the temporary MP3 file.
    """
    tts = gTTS(story_text, lang="en")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name

# ---------- Streamlit User Interface ----------

def main():
    # Page configuration
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
            # 1. Image to caption
            caption = img2text(image)
            st.subheader("📝 What I See")
            st.write(caption)

            # 2. Caption to story
            story = text2story(caption)
            st.subheader("📚 Your Story")
            st.write(story)

            # 3. Story to audio
            audio_file = text2audio(story)
            st.subheader("🔊 Listen to Your Story")
            st.audio(audio_file, format="audio/mp3")

if __name__ == "__main__":
    main()
