import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import pandas as pd
import base64

# Function to encode the image
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Background image path
background_image_path = r"P:\Study\Intership\AICTE Intership\Edunet SHELL Internship\Garbage_classification\red.jpg"

# Generate base64 string
base64_bg = get_base64_image(background_image_path)

# Apply dark + red themed CSS
page_bg_css = f"""
<style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{base64_bg}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #FFCCCC;
        font-family: 'Arial', sans-serif;
    }}

    h1, h2, h3, h4, h5, h6 {{
        color: #FF4444 !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.6);
    }}

    .stSidebar {{
        background-color: #1A1A1A;
        color: #FFCCCC !important;
    }}

    .stButton>button {{
        background-color: #B22222;  /* Firebrick */
        color: #FFFFFF !important;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }}

    .stButton>button:hover {{
        background-color: #8B0000;  /* Dark red */
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }}

    .stFileUploader {{
        background-color: #2A2A2A;
        border-radius: 10px;
        padding: 15px;
        border: 2px dashed #FF4444;
    }}

    .stFileUploader:hover {{
        border-color: #FF6F61;
    }}

    .prediction-box {{
        background-color: #2A2A2A;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 10px rgba(255, 69, 0, 0.5);
        animation: fadeIn 0.5s ease-in;
    }}

    @keyframes fadeIn {{
        0% {{ opacity: 0; }}
        100% {{ opacity: 1; }}
    }}

    .stDownloadButton>button {{
        background-color: #DC143C;  /* Crimson */
        color: #FFFFFF !important;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
        border: none;
    }}

    .stDownloadButton>button:hover {{
        background-color: #A80000;
        transform: translateY(-2px);
    }}

    .stProgress .st-bo {{
        background-color: #FF4444;
    }}
</style>
"""
st.markdown(page_bg_css, unsafe_allow_html=True)

# Class names
CLASS_NAMES = ['carboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
IMG_SIZE = (224, 224)

# Session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

st.title("♻️ Garbage Classification System")
st.markdown("Upload an image of trash to classify it into one of the categories: **cardboard, glass, metal, paper, plastic, or trash.**")

# Load model
@st.cache_resource
def load_trained_model():
    try:
        return load_model(r'P:\Study\Intership\AICTE Intership\Edunet SHELL Internship\Garbage_classification\model231.h5')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

with st.spinner('Loading model...'):
    model = load_trained_model()

if not model:
    st.stop()

# Uploader
uploaded_file = st.file_uploader("📷 Choose an image file (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

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

    if st.button("🔍 Predict & Save"):
        with st.spinner("Classifying..."):
            img_array = preprocess_image(image)
            raw_predictions = model.predict(img_array)
            adjusted_predictions = temperature_scaled_softmax(raw_predictions, temp=0.5)
            predicted_class_index = np.argmax(adjusted_predictions, axis=1)[0]
            predicted_label = CLASS_NAMES[predicted_class_index]
            confidence = np.max(adjusted_predictions) * 100

            st.session_state.predictions.append({
                'Image': uploaded_file.name,
                'Prediction': predicted_label,
                'Confidence': f"{confidence:.2f}%"
            })

            st.markdown(
                f"""
                <div class="prediction-box">
                    <h3>Prediction Result</h3>
                    <p><strong>Category:</strong> {predicted_label}</p>
                    <p><strong>Confidence:</strong> {confidence:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("**Class Probabilities:**")
            prob_data = {CLASS_NAMES[i]: round(prob * 100, 2) for i, prob in enumerate(adjusted_predictions[0])}
            prob_df = pd.DataFrame(list(prob_data.items()), columns=['Class', 'Probability (%)'])
            st.bar_chart(prob_df.set_index('Class'))

# Prediction history
if st.session_state.predictions:
    st.markdown("### Prediction History")
    result_df = pd.DataFrame(st.session_state.predictions)
    st.dataframe(result_df, use_container_width=True)

    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=csv,
        file_name='predictions.csv',
        mime='text/csv'
    )

# Clear button
if st.button("🗑️ Clear All"):
    st.session_state.predictions = []
    st.experimental_rerun()
