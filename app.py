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

# Sidebar with Project Details (Unique UI/UX Styling)
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


# Title and Description
st.title(':orange[EmoNet 😎] - Advanced Emotion Detector')
st.write("Welcome to **EmoNet**, a real-time AI-powered emotion detection system. Choose your input method below!")

# Input Method Section with Reduced Border Size and More Spacing
st.markdown("""
    <div style="
        padding: 0px; 
        border: 1.5px solid #28a745; 
        border-radius: 10px; 
        background-color: #f8f9fa;
        margin-bottom: 10px;
        ">
        <h3 style="color: #28a745;">🎥 Choose Input Method</h3>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

# Mode selection
mode = st.radio("Select an Input Method:", ("📷 Use Webcam", "📤 Upload an Image"))

if mode == "📷 Use Webcam":
    st.subheader("🎥 Live Emotion Detection")
    webcam = cv2.VideoCapture(0)
    FRAME_WINDOW = st.image([])
    run = st.checkbox('Start Webcam')

    while run:
        ret, frame = webcam.read()
        if not ret:
            st.error("Failed to capture video")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (48, 48))
            img = extract_features(face_resized)

            pred = model.predict(img)
            emotion, emoji = labels[np.argmax(pred)]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"{emotion} {emoji}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    webcam.release()
    cv2.destroyAllWindows()

elif mode == "📤 Upload an Image":
    st.subheader("🖼 Upload an Image")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display uploaded image in the center
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
            st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert image to OpenCV format
        img_array = np.array(image)
        face_detected = False

        # Detect faces
        faces = face_cascade.detectMultiScale(img_array, 1.3, 5)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_detected = True
                face = img_array[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (48, 48))
                img = extract_features(face_resized)

                pred = model.predict(img)
                emotion, emoji = labels[np.argmax(pred)]

                # Beautiful output box
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

        if not face_detected:
            st.warning("⚠ No face detected in the uploaded image.")

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
            <li><b>Model:</b> rNN (Recurrent Neural Network)</li>
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