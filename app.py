import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# --- 1. CONFIGURATION ---
# Use the full Google Drive "Share" link - gdown now handles this automatically!
DRIVE_URL = 'https://drive.google.com/file/d/1aAMq0drxZzIMqDxX7DK7ZyOOX9Dbndz7/view?usp=sharing'
MODEL_PATH = "brain_tumor_model.h5"

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_model_from_drive():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI Model from Google Drive..."):
            try:
                # In v6.0+, just pass the URL. 'fuzzy' and 'quiet' are removed.
                gdown.download(url=DRIVE_URL, output=MODEL_PATH)
            except Exception as e:
                st.error(f"Download Error: {e}")
                st.stop()
    
    # Verification: Ensure it's not an empty file or HTML error page
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) < 1000000:
        os.remove(MODEL_PATH)
        st.error("Download failed. Please ensure the Google Drive link is set to 'Anyone with the link'!")
        st.stop()
        
    return tf.keras.models.load_model(MODEL_PATH)

# Initialize the model
model = load_model_from_drive()

# --- 3. PREDICTION ENGINE ---
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

def predict(image):
    img = image.resize((150, 150))
    img_array = np.array(img)
    if img_array.shape[-1] == 4: # Handle PNG transparency
        img_array = img_array[:, :, :3]
    
    img_input = img_array.reshape(1, 150, 150, 3) / 255.0
    preds = model.predict(img_input)
    return labels[np.argmax(preds)], preds

# --- 4. USER INTERFACE ---
st.set_page_config(page_title="NeuroGuard AI", page_icon="🧠")
st.title("🧠 NeuroGuard: MRI Analysis")
st.write("Professional-grade brain tumor classification using Deep Learning.")

uploaded_file = st.file_uploader("Upload MRI Scan (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Target Scan", use_container_width=True)
    
    with st.status("Analyzing Scan..."):
        label, raw_preds = predict(img)
    
    st.subheader(f"Result: :red[{label.replace('_', ' ').title()}]")
    
    # Display confidence for judges
    cols = st.columns(4)
    for i, col in enumerate(cols):
        col.metric(labels[i].title(), f"{raw_preds[0][i]*100:.1f}%")
