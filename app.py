import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os
import gdown

# --- 1. CONFIGURATION (Update your ID here) ---
FILE_ID = 'PASTE_YOUR_GOOGLE_DRIVE_ID_HERE' 
MODEL_PATH = "brain_tumor_model.h5"
# This URL format bypasses the "Large File" warning screen
URL = f'https://drive.google.com/uc?id="1aAMq0drxZzIMqDxX7DK7ZyOOX9Dbndz7"&confirm=t'

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_model_from_drive():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive... This may take a minute."):
            # fuzzy=True helps handle redirection/confirmations
            gdown.download(URL, MODEL_PATH, quiet=False, fuzzy=True)
    
    # Safety Check: If the file is tiny, the download failed
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) < 1000000:
        os.remove(MODEL_PATH)
        st.error("Download failed or link is restricted. Please check Google Drive 'Anyone with link' settings!")
        st.stop()
        
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model_from_drive()

# --- 3. PREDICTION LOGIC ---
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

def predict(image):
    # Match the preprocessing to your training (150x150)
    img = image.resize((150, 150))
    img_array = np.array(img)
    
    # Ensure 3 channels (RGB)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
        
    img_input = img_array.reshape(1, 150, 150, 3)
    # Normalization (If you used 1/255 in training)
    img_input = img_input / 255.0
    
    preds = model.predict(img_input)
    return labels[np.argmax(preds)], preds

# --- 4. STREAMLIT UI ---
st.title("🧠 NeuroGuard: Brain Tumor Classification")
st.write("Upload an MRI scan to identify tumor types and see AI confidence.")

uploaded_file = st.file_report("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Scan', use_column_width=True)
    
    label, raw_preds = predict(image)
    
    st.subheader(f"Diagnosis: **{label.replace('_', ' ').title()}**")
    st.write(f"Confidence Scores: {raw_preds}")
