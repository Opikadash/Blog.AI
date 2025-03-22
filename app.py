import os
import logging
import streamlit as st
import torch
import asyncio 
from diffusers import StableDiffusionPipeline
from PIL import Image
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load API Keys
google_gemini_api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")

if not google_gemini_api_key:
    st.error("Please set the GOOGLE_GEMINI_API_KEY environment variable.")
    st.stop()

if not huggingface_api_key:
    st.error("Please set the HUGGINGFACE_API_KEY environment variable.")
    st.stop()

# Configure Google Gemini AI
genai.configure(api_key=google_gemini_api_key)

# Create cache directory for images
CACHE_DIR = "generated_images"
os.makedirs(CACHE_DIR, exist_ok=True)

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Load Stable Diffusion model with dtype fix
@st.cache_resource  # Cache the model to avoid reloading
def load_model():
    """Load Stable Diffusion model from Hugging Face."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32 

        model = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=dtype,
            token=huggingface_api_key
        ).to(device)

        logging.info("✅ Stable Diffusion model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"❌ Error loading model: {e}")
        return None

stable_diffusion_model = load_model()

# Image generation 
def generate_image(prompt):
    """Generate an image using Stable Diffusion with optimizations."""
    sanitized_prompt = prompt.replace(" ", "_")
    image_path = os.path.join(CACHE_DIR, f"{sanitized_prompt}.png")

    if os.path.exists(image_path):
        try:
            Image.open(image_path).verify()
            logging.info(f"✅ Using cached image for: {prompt}")
            return image_path
        except Exception:
            logging.warning(f"⚠️ Corrupt cached image detected: {image_path}. Regenerating...")

    if stable_diffusion_model is None:
        logging.error("❌ Model not loaded, skipping image generation.")
        return None

    try:
        logging.info(f"🎨 Generating image for: {prompt}")

        image = stable_diffusion_model(
            prompt,
            height=384,  # Lower resolution
            width=384,
            num_inference_steps=30  # Fewer steps
        ).images[0]

        image.save(image_path)
        return image_path
    except Exception as e:
        logging.error(f"❌ Error generating image: {e}")
        return None

# Blog content generation function
def generate_blog(blog_title, keywords, num_words, num_images):
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    prompt = f"""
    Generate a well-structured, unique, creative and original blog post with title: "{blog_title}".
    Incorporate these keywords: {keywords}.
    The post should be approximately {num_words} words long.
    Include {num_images} image placeholders using <IMAGE: description> format.
    """
    response = model.generate_content(prompt)
    if not response or not response.text:
        return "No response from Gemini AI."

    blog_text = response.text
    parts = blog_text.split("<IMAGE:")
    processed_text = [parts[0]]

    for i in range(1, len(parts)):
        remaining = parts[i].split(">", 1)
        if len(remaining) == 2:
            description = remaining[0].strip()
            image_path = generate_image(description)
            if image_path:
                processed_text.append(f"\n![{description}]({image_path})\n")
            else:
                processed_text.append(f" (Image failed: {description})\n")
            processed_text.append(remaining[1])
        else:
            processed_text.append("<IMAGE:" + parts[i])

    return "".join(processed_text)

#Front-end UI using streamlit
st.title("📝 Blog.AI : Your AI Blog Companion")
st.markdown("""
    **Struggling to write? Let AI do it for you!** ✍️
    
""")

with st.sidebar:
    st.title("Blog Configuration")
    blog_title = st.text_input("📌 Blog Title")
    keywords = st.text_area("🔑 Keywords (comma-separated)")
    num_words = st.slider("📝 Number of words", min_value=250, max_value=1500, step=250)
    num_images = st.number_input("🖼 Number of Images", min_value=1, max_value=5, step=1)
    submit_button = st.button("Generate Blog")

if submit_button:
    if not blog_title or not keywords:
        st.warning("⚠️ Please enter a blog title and keywords.")
    else:
        with st.spinner("✍️ AI is writing your blog..."):
            blog_post = generate_blog(blog_title, keywords, num_words, num_images)

        st.markdown("""
        <style>
        @keyframes typing {
            from { width: 0 }
            to { width: 100% }
        }
        .typing-effect::after {
            content: '|';
            display: inline-block;
            animation: blink 0.8s infinite;
        }
        @keyframes blink {
            50% { opacity: 0 }
        }
        </style>
        <p class='typing-effect'>Your blog is ready! ✍️</p>
        """, unsafe_allow_html=True)

        if blog_post:
            st.header("📄Get,set,go!")
            for line in blog_post.split("\n"):
                if "![" in line and "](" in line:
                    parts = line.split("![", 1)[1].split("](", 1)
                    description = parts[0]
                    image_path = parts[1][:-1]
                    st.image(image_path, caption=description)
                else:
                    st.write(line)

            st.download_button("📥 Download Blog", blog_post, file_name="generated_blog.txt")
            st.success("✅ Blog is ready to post!")
        else:
            st.error("❌ Try Again.")

