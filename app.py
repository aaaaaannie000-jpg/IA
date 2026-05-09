"""
Storytelling Application for Kids (Ages 3-10)
=============================================
- Upload an image
- Generate a caption using BLIP image-to-text
- Expand the caption into a 50-100 word children's story with distilgpt2
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

# ---------- Model Caching ----------

@st.cache_resource
def load_image_caption_model():
    """Load BLIP image captioning model."""
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_story_generator_model():
    """Load distilgpt2 for story generation."""
    return pipeline("text-generation", model="distilgpt2")

# ---------- Core Functions ----------

def img2text(image: Image.Image) -> str:
    """Generate a descriptive caption from the uploaded image."""
    captioner = load_image_caption_model()
    return captioner(image)[0]["generated_text"]

def text2story(caption: str) -> str:
    """
    Turn the image caption into a short, child-friendly story (50-100 words)
    using a structured prompt.
    """
    generator = load_story_generator_model()

    # Your provided prompt template
    prompt = (
        f'You are a creative storyteller for children aged 3-10. Based on the image description provided below, write a short, fun, and imaginative story.\n\n'
        f'**Image Description:**\n'
        f'"{caption}"\n\n'
        f'**Guidelines:**\n'
        f'1.  **Theme & Relevance:** The story must directly include the main subject and action from the image description above. Do not introduce unrelated elements.\n'
        f'2.  **Target Audience:** Use simple, easy-to-understand language suitable for young kids. The tone should be positive, magical, or educational.\n'
        f'3.  **Length:** Strictly keep the story between 50 to 100 words.\n'
        f'4.  **Content:** Focus on feelings, colors, or a small adventure. Avoid complex plots or scary themes.\n\n'
        f'Story:'
    )

    result = generator(
        prompt,
        max_new_tokens=150,
        temperature=0.9,
        pad_token_id=generator.tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2
    )[0]["generated_text"]

    # Extract only the generated story (after "Story:")
    return result[len(prompt):].strip()

def text2audio(story_text: str) -> str:
    """Convert story text to speech and return path to temporary MP3 file."""
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

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Your Picture", use_container_width=True)

        with st.spinner("✨ Creating your story... This may take a few seconds."):
            # Step 1: Image → Caption
            caption = img2text(image)
            st.subheader("📝 What I See")
            st.write(caption)

            # Step 2: Caption → Story
            story = text2story(caption)
            st.subheader("📚 Your Story")
            st.write(story)

            # Step 3: Story → Audio
            audio_file = text2audio(story)
            st.subheader("🔊 Listen to Your Story")
            st.audio(audio_file, format="audio/mp3")

if __name__ == "__main__":
    main()
