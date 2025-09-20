# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# ========== CONFIG ==========
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "mobilenetv2_pepaya.keras")  # pastikan file model ada
ALLOWED = {"png", "jpg", "jpeg"}

st.set_page_config(
    page_title="Pepaya Classifier", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ========== CSS CUSTOM (berdasarkan desain HTML Anda) ==========
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;600;700&display=swap');
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}
    
    /* Reset and Base Styles */
    .stApp {
        background: linear-gradient(135deg, #f9f9f9, #d2fbd2);
        font-family: "Segoe UI", Arial, sans-serif;
    }
    
    /* Main Container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 450px;
    }
    
    /* Custom Container */
    .pepaya-container {
        background: #fff;
        padding: 30px 25px;
        border-radius: 16px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
        text-align: center;
        animation: fadeIn 0.8s ease;
        margin: 0 auto;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Title Styling */
    .pepaya-title {
        font-size: 1.8rem;
        margin-bottom: 20px;
        color: #2c3e50;
        font-weight: 700;
        text-align: center;
    }
    
    /* Button Styling */
    .stButton > button {
        width: 100% !important;
        margin: 8px 0 !important;
        padding: 12px 20px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 30px !important;
        cursor: pointer !important;
        background: linear-gradient(135deg, #28a745, #4cd964) !important;
        color: white !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(40, 167, 69, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(40, 167, 69, 0.4) !important;
    }
    
    .stButton > button:focus {
        box-shadow: 0 5px 15px rgba(40, 167, 69, 0.4) !important;
    }
    
    /* File Uploader Styling */
    .uploadedFile {
        border-radius: 12px !important;
        overflow: hidden;
    }
    
    /* Image Preview */
    .stImage {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.15);
        margin: 15px 0;
    }
    
    .stImage > img {
        border-radius: 12px;
    }
    
    /* Result Styling */
    .result-success {
        margin-top: 20px;
        font-size: 1.1rem;
        font-weight: bold;
        color: #2c3e50;
        padding: 15px;
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border-radius: 12px;
        border: 1px solid #c3e6cb;
    }
    
    .result-error {
        margin-top: 20px;
        font-size: 1.1rem;
        font-weight: bold;
        color: #721c24;
        padding: 15px;
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border-radius: 12px;
        border: 1px solid #f5c6cb;
    }
    
    .result-info {
        margin-top: 20px;
        font-size: 1rem;
        color: #0c5460;
        padding: 15px;
        background: linear-gradient(135deg, #d1ecf1, #bee5eb);
        border-radius: 12px;
        border: 1px solid #bee5eb;
    }
    
    /* Spinner */
    .stSpinner {
        text-align: center;
        margin: 20px 0;
    }
    
    /* Hide file uploader label */
    .stFileUploader label {
        display: none !important;
    }
    
    /* Camera input styling */
    .stCameraInput label {
        display: none !important;
    }
    
    /* Custom spacing */
    .element-container {
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ========== LOAD MODEL ==========
@st.cache_resource
def load_model(path):
    try:
        return tf.keras.models.load_model(path)
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None

model = load_model(MODEL_PATH)

if model is None:
    st.error("❌ Gagal memuat model. Pastikan file `mobilenetv2_pepaya.keras` ada di direktori yang sama.")
    st.stop()

# Label classes
num_classes = model.output_shape[-1] if hasattr(model, "output_shape") else 3
all_labels = ["matang", "mentah", "setengah"]
class_labels = all_labels[:num_classes] if num_classes else all_labels

# ========== UTILS ==========
def predict_image_bytes(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224))
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, 0).astype(np.float32)
        preds = model.predict(arr, verbose=0)
        idx = int(np.argmax(preds, axis=1)[0])
        label = class_labels[idx]
        conf = float(np.max(preds))
        return label, conf, None
    except Exception as e:
        return None, None, str(e)

# ========== MAIN UI ==========
# Container wrapper
st.markdown('<div class="pepaya-container">', unsafe_allow_html=True)

# Title
st.markdown('<h1 class="pepaya-title">🍈 Pepaya Maturity Classifier</h1>', unsafe_allow_html=True)

# Initialize session state
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'current_image' not in st.session_state:
    st.session_state.current_image = None

# Camera Input
camera_file = None
if hasattr(st, 'camera_input'):
    camera_file = st.camera_input("📷 Ambil Foto")

# File Upload
uploaded_file = st.file_uploader("🖼 Pilih dari Galeri", type=list(ALLOWED))

# Determine which file to use
current_file = camera_file if camera_file is not None else uploaded_file

# Display image if available
if current_file is not None:
    try:
        file_bytes = current_file.getvalue() if hasattr(current_file, 'getvalue') else current_file.read()
        st.session_state.current_image = file_bytes
        st.image(file_bytes, caption="Preview Gambar", use_column_width=True)
    except Exception as e:
        st.markdown(f'<div class="result-error">❌ Error membaca file: {e}</div>', unsafe_allow_html=True)

# Prediction button
if st.button("🔍 Prediksi"):
    if st.session_state.current_image is not None:
        with st.spinner("⏳ Memproses..."):
            label, conf, error = predict_image_bytes(st.session_state.current_image)
            
            if error:
                st.session_state.prediction_result = f'<div class="result-error">❌ Error prediksi: {error}</div>'
            else:
                confidence_percent = conf * 100
                result_text = f"✅ Hasil: {label.upper()} ({confidence_percent:.2f}%)"
                st.session_state.prediction_result = f'<div class="result-success">{result_text}</div>'
    else:
        st.session_state.prediction_result = '<div class="result-info">⚠️ Pilih atau ambil foto terlebih dahulu!</div>'

# Display prediction result
if st.session_state.prediction_result:
    st.markdown(st.session_state.prediction_result, unsafe_allow_html=True)
elif st.session_state.current_image is None:
    st.markdown('<div class="result-info">📱 Pilih atau ambil foto pepaya untuk memulai klasifikasi tingkat kematangan.</div>', unsafe_allow_html=True)

# Close container
st.markdown('</div>', unsafe_allow_html=True)

# Add some spacing at the bottom
st.markdown('<br><br>', unsafe_allow_html=True)