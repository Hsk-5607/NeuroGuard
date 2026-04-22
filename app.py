import os
import tensorflow as tf
import streamlit as st
import gdown

# 1. Configuration
FILE_ID = '1A2B3C4D5E6F7G8H9' # <--- PASTE YOUR ID HERE
MODEL_PATH = "brain_tumor_model.h5"
URL = f'https://drive.google.com/uc?id={FILE_ID}'

# 2. Function to fetch and load the model
@st.cache_resource
def load_model_from_drive():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Fetching model from Google Drive..."):
            # This downloads the file directly into the server
            gdown.download(URL, MODEL_PATH, quiet=False)
    
    return tf.keras.models.load_model(MODEL_PATH)

# 3. Load it
model = load_model_from_drive()
