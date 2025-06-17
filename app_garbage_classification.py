import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import pandas as pd
import base64
import tensorflow as tf

# Function to encode the image
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Path to your background image (update as needed)
background_image_path = r'C:\Users\sksan\OneDrive\Desktop\infosys intern\New folder (2)\background.jpg'

# Generate base64 string
base64_bg = get_base64_image(background_image_path)

# Apply CSS to set the background and text color
page_bg_css = f"""
<style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{base64_bg}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #FFFFFF;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: #FFFFFF !important;
    }}
    .stSidebar {{
        color: #FFFFFF !important;
    }}
    .stButton>button {{
        color: black !important;
        background-color: #FFFFFF !important;
    }}
</style>
"""
st.markdown(page_bg_css, unsafe_allow_html=True)

# Define class names for garbage classification
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
IMG_SIZE = (224, 224)

st.title("GarbageSort: AI-Powered Waste Classification")
st.write("Upload an image of a waste item to classify it into one of the categories: cardboard, glass, metal, paper, plastic, or trash.")

# Cache model loading
@st.cache_resource
def load_trained_model():
    try:
        return load_model(r'C:\Users\sksan\OneDrive\Desktop\infosys intern\New folder (2)\garbage_classification_model.h5')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

with st.spinner('Loading model...'):
    model = load_trained_model()

if not model:
    st.stop()

# Initialize session state
if 'items' not in st.session_state:
    st.session_state.items = []

# Sidebar: Item information
st.sidebar.header("Item Information")
item_id = st.sidebar.text_input("Item ID")
item_description = st.sidebar.text_input("Item Description")

# File uploader
uploaded_file = st.file_uploader("Choose an image file (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def temperature_scaled_softmax(logits, temp=0.5):
    exp_scaled = np.exp(logits / temp)
    return exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=False, width=300)

    if st.button("Predict & Save Data"):
        img_array = preprocess_image(image)
        raw_predictions = model.predict(img_array)
        adjusted_predictions = temperature_scaled_softmax(raw_predictions, temp=0.5)
        predicted_class_index = np.argmax(adjusted_predictions, axis=1)[0]
        predicted_label = CLASS_NAMES[predicted_class_index]
        confidence = np.max(adjusted_predictions) * 100

        # Debugging: Show probability distribution
        st.write("Adjusted Class Probabilities:", {CLASS_NAMES[i]: round(prob * 100, 2) for i, prob in enumerate(adjusted_predictions[0])})

        # Save item data
        st.session_state.items.append({
            "Item ID": item_id,
            "Item Description": item_description,
            "Waste Category": predicted_label.upper(),
            "Confidence (%)": round(confidence, 2)
        })

        st.success(f"Prediction: **{predicted_label}** with {confidence:.2f}% confidence")

# Button to clear all fields
if st.button("Clear All Fields"):
    st.session_state.items.clear()

# Display item data
if st.session_state.items:
    st.write("## Prediction and Item Information:")
    result_df = pd.DataFrame(st.session_state.items)
    st.table(result_df)

    # Download predictions as CSV
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions as CSV", data=csv, file_name='garbage_predictions.csv', mime='text/csv')