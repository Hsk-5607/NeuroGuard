import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Page Setup
st.set_page_config(page_title="NeuroGuard AI", layout="centered")

st.title("🧠 NeuroGuard: Brain Tumor Classifier")

# Check if model exists
model_path = 'brain_tumor_model.h5'

@st.cache_resource
def load_model():
    if os.path.exists(model_path):
        # compile=False avoids errors with custom optimizers/metrics from training
        return tf.keras.models.load_model(model_path, compile=False)
    return None

model = load_model()

if model is None:
    st.error(f"❌ Model file '{model_path}' not found! Please upload it to your Space.")
else:
    st.write("Upload an MRI image (JPG/PNG) to classify.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded MRI Scan', use_container_width=True)
        
        # Preprocessing: 150x150 to match EfficientNetB0 training
        img = image.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        with st.spinner('Analyzing...'):
            prediction = model.predict(img_array)
            classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
            result = classes[np.argmax(prediction)]
            conf = np.max(prediction) * 100
            
            st.success(f"**Prediction: {result}**")
            st.info(f"**Confidence: {conf:.2f}%**")