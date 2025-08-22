# libraries
import streamlit as st
from fastai.vision.all import *
import torch
import pathlib

# Fix WindowsPath issue if model was trained on Linux/Colab
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# --- Title & description ---
st.title("✨\"Meva\" Classifier")
st.title("🍎🍊🍇🥕🍉🫑🍅")
st.write("This app classifies images of fruits and vegetables!")

# --- Load model once (cached to avoid reloading every run) ---
@st.cache_resource
def load_model():
    return load_learner('meva_model.pkl')

model = load_model()

# --- Upload image ---
file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

if file is not None:
    # Show uploaded image
    st.image(file, caption='Uploaded Image', use_container_width=True)

    # Convert to PIL image
    img = PILImage.create(file)

    # --- Prediction ---
    pred, pred_id, probs = model.predict(img)

    max_prob = torch.max(probs)
    if max_prob < 0.5:
        st.warning("⚠️ Unknown image. Please try another one!")
    else:
        st.success(f"✅ Prediction: {pred}")
        st.info(f"📊 Probability: {max_prob*100:.2f}%")

# --- Footer ---
st.write("---")
st.write("⚡Developed by [Elyorbek Akhmatov](https://www.linkedin.com/in/elyorakhmat/)")