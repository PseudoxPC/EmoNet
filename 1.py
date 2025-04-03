import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model("emotiondetector.h5")

# Load face detection model
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to preprocess image for prediction
def extract_features(image):
    feature = np.array(image).reshape(1, 48, 48, 1)
    return feature / 255.0  # Normalize

# Emotion labels with emojis
labels = {
    0: ('Angry', '😠'),
    1: ('Disgust', '🤢'),
    2: ('Fear', '😨'),
    3: ('Happy', '😃'),
    4: ('Neutral', '😐'),
    5: ('Sad', '😢'),
    6: ('Surprise', '😲')
}

# Streamlit UI
st.set_page_config(page_title='EmoNet 😎', layout='wide')

# Sidebar with project details
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; font-size: 22px; font-weight: bold; color: #ffcc00;">
            *EmoNet: Emotion Detector* 
        </div>
        <hr>
        <div style="color: white; font-size: 16px; background: linear-gradient(to right, #ff0080, #ffcc00); padding: 10px; border-radius: 10px;">
            <b>Developed By:</b><br>
            Krishna Gupta (0901AD221044) <br>
            Priyanshu Chouhan (0901AD221058) <br>
        </div>
        <br>
        <div style="color: white; font-size: 16px; background: linear-gradient(to right, #00c6ff, #0072ff); padding: 10px; border-radius: 10px;">
            <b>Guided By:</b><br>
            Dr. Tej Singh 🎓
        </div>
        <br>
        <div style="text-align: center; font-size: 16px;">
            🤖 Built with OpenCV | TensorFlow | Streamlit
        </div>
        <hr>
    """, unsafe_allow_html=True)

# Main App Title
st.title(':orange[EmoNet 😎] - Advanced Emotion Detector')
st.subheader("Detect human emotions using Deep Learning and Computer Vision.")

# Input Method Section with Reduced Border Size and More Spacing
st.markdown("""
    <div style="
        padding: 15px; 
        border: 1.5px solid #28a745; 
        border-radius: 10px; 
        background-color: #f8f9fa;
        margin-bottom: 20px;
        ">
        <h3 style="color: #28a745;">🎥 Choose Input Method</h3>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

# Webcam Input
with col1:
    st.markdown("<h4>📷 Use Webcam</h4>", unsafe_allow_html=True)
    webcam_image = st.camera_input("Capture Image", label_visibility='collapsed')

    if webcam_image is not None:
        image = Image.open(webcam_image).convert("L")
        img_array = np.array(image)

        # Detect face
        faces = face_cascade.detectMultiScale(img_array, 1.3, 5)
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face = img_array[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (48, 48))
                img = extract_features(face_resized)
                pred = model.predict(img)
                emotion, emoji = labels[np.argmax(pred)]
                
                # Output Box Styling
                st.markdown(f"""
                <div style="
                    background-color: #f0f0f0; 
                    padding: 15px; 
                    border: 1.5px solid #ff5733; 
                    border-radius: 10px;
                    text-align: center;
                    font-size: 20px;
                    font-weight: bold;
                    color: black;
                    ">
                    Emotion Detected: <span style="color: #FF5733;">{emotion} {emoji}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No face detected in the captured image.")

# File Upload Input
with col2:
    st.markdown("<h4>📤 Upload an Image</h4>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drag and Drop Image Here", type=["jpg", "png", "jpeg"], label_visibility='collapsed')
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        img_array = np.array(image)

        # Detect face
        faces = face_cascade.detectMultiScale(img_array, 1.3, 5)
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face = img_array[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (48, 48))
                img = extract_features(face_resized)
                pred = model.predict(img)
                emotion, emoji = labels[np.argmax(pred)]

                # Output Box Styling
                st.markdown(f"""
                <div style="
                    background-color: #f0f0f0; 
                    padding: 15px; 
                    border: 1.5px solid #ff5733; 
                    border-radius: 10px;
                    text-align: center;
                    font-size: 20px;
                    font-weight: bold;
                    color: black;
                    ">
                    Emotion Detected: <span style="color: #FF5733;">{emotion} {emoji}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No face detected in the uploaded image.")

st.markdown("""
    <div style="
        padding: 15px; 
        border: 1.5px solid #007bff; 
        border-radius: 10px; 
        background-color: #000000;
        margin-top: 20px;
        ">
        <h3 style="color: #007bff;">🔍 About the Model</h3>
        <p style="font-size: 16px;">This model is based on CNN and trained on the FER-2013 dataset for emotion recognition.</p>
        <ul style="font-size: 16px;">
            <li><b>Model:</b> CNN (Convolutional Neural Network)</li>
            <li><b>Architecture:</b> 4 Conv Layers + 2 Fully Connected Layers</li>
            <li><b>Dataset:</b> FER-2013 (Facial Expression Recognition)</li>
            <li><b>Accuracy:</b> 72% (Validation), 75% (Training)</li>
            <li><b>Loss Function:</b> Categorical Crossentropy</li>
            <li><b>Optimizer:</b> Adam Optimizer</li>
            <li><b>Input Size:</b> 48x48 Grayscale Facial Images</li>
            <li><b>Frameworks Used:</b> TensorFlow, Keras</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

st.markdown("<div style='text-align: center; color: gray; font-size: 14px;'>🚀 Project by Priyanshu & Krishna | Powered by Deep Learning & Computer Vision</div>", unsafe_allow_html=True)