# Blog.AI
LLM using API tokens
# Blog.AI: AI-Powered Blog Generator 🚀

## Overview 📖
Blog.AI is a **Streamlit-based AI blog generator** that utilizes **Google Gemini AI** for text generation and **Stable Diffusion** for AI-generated images. It allows users to create unique, creative, and structured blog posts with AI-generated images based on their inputs.

## Features ✨
- 📝 **AI-Powered Blog Writing** using Google Gemini AI
- 🎨 **AI-Generated Images** using Stable Diffusion
- 📥 **Download Generated Blog** as a text file
- 🖥️ **Simple & Interactive UI** using Streamlit

## Folder Structure 📂
```
.
├── .venv/                  # Virtual environment (ignored in Git)
├── generated_images/       # Folder where generated images are stored
├── app.py                  # Main Streamlit application script
├── diffusion_pytorch_model.safetensors  # Stable Diffusion model file
├── README.md               # Project documentation (this file)
└── requirements.txt        # Python dependencies
```

## Installation 🛠️

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/yourusername/blog-ai.git
cd blog-ai
```

### 2️⃣ Create a Virtual Environment
```sh
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
.venv\Scripts\activate     # On Windows
```

### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 4️⃣ Set Up API Keys
Before running the application, set the required API keys as environment variables:
```sh
export GOOGLE_GEMINI_API_KEY="your_google_gemini_api_key"
export HUGGINGFACE_API_KEY="your_huggingface_api_key"
```
On Windows (PowerShell):
```powershell
$env:GOOGLE_GEMINI_API_KEY="your_google_gemini_api_key"
$env:HUGGINGFACE_API_KEY="your_huggingface_api_key"
```

## Usage 🚀
Run the Streamlit app:
```sh
streamlit run app.py
```
After running the command, open **http://localhost:8501/** in your browser.

## Deploying on Cloud GPU ☁️
If you want to run Blog.AI on a **Cloud GPU** (Google Colab, RunPod, Paperspace, Lambda Labs):
1. Upload `app.py` to your cloud instance.
2. Install dependencies using `pip install -r requirements.txt`
3. Make sure CUDA is available (`torch.cuda.is_available()` should return `True`).
4. Run the app using `streamlit run app.py --server.port 8501`

## Contributing 🤝
Feel free to open issues and pull requests to improve this project!

## License 📝
This project is licensed under the MIT License.

---


