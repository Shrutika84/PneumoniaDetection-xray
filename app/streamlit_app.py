import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import sys

# Add src to path to allow module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from model_cnn import get_model

# --- Page Config ---
st.set_page_config(page_title="Pneumonia Detection", layout="centered")
st.title("🩻 Pneumonia Detection from Chest X-Rays")
st.markdown("Upload a chest X-ray image to detect **Pneumonia** or **Normal**.")

# --- Constants ---
MODEL_PATH = 'outputs/best_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load Model ---
@st.cache_resource
def load_model():
    model = get_model(model_name='efficientnet', pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# --- Image Preprocessing ---
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(DEVICE)

# --- Prediction ---
def predict(image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item()
    return predicted.item(), confidence

# --- Upload UI ---
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")  # Ensure 3 channels
    st.image(image, caption='Uploaded Image', use_column_width=True)

    with st.spinner("🔍 Classifying..."):
        input_tensor = preprocess_image(image)
        pred_class, confidence = predict(input_tensor)
        label = "Pneumonia" if pred_class == 1 else "Normal"

        st.success(f"**Prediction:** {label}")
        st.info(f"**Confidence:** {confidence * 100:.2f}%")
