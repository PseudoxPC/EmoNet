<!-- Banner -->
<h1 align="center">🤖 EmoNet - Facial Expression Detection using CNN</h1>
<p align="center">
  <b>Real-time Emotion Detection powered by Deep Learning (FER-2013 Dataset)</b>  
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow" />
  <img src="https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
</p>

---

## 📌 Overview  
EmoNet is a **deep learning-based emotion detection system** that can identify human facial expressions in **real-time** using a **Convolutional Neural Network (CNN)**.  
The project is deployed with **Streamlit** and supports both **webcam-based live detection** and **image uploads**.  

---

## ✨ Features  
- 🎥 **Real-time emotion detection** using webcam  
- 🖼 **Image upload support** for offline testing  
- 🧠 **Trained CNN model** on FER-2013 dataset  
- 📊 Supports **7 emotions**: Angry 😠, Disgust 🤢, Fear 😨, Happy 😃, Neutral 😐, Sad 😢, Surprise 😲  
- 🖥 Interactive UI built with **Streamlit**  
- ⚡ Lightweight & easy-to-run with just a few dependencies  

---

## 🛠️ Tech Stack  
**Languages & Frameworks:**  
- Python   
- TensorFlow & Keras  
- OpenCV  
- Streamlit  

**Dataset:**  
- [FER-2013 (Facial Expression Recognition Dataset)](https://www.kaggle.com/datasets/msambare/fer2013)  

---

## 📂 Project Structure  
# EmoNet/
- │-- app.py # Streamlit app (main UI + logic)
- │-- emotiondetector.h5 # Trained CNN model
- │-- emotiondetector.json # Model architecture
- │-- requirements.txt # Project dependencies
- │-- sample_images/ # Example images (for testing)
- │-- README.md # Project documentation


---

## 🚀 Installation & Usage  

### 🔧 Setup Environment  

# Clone the repository
- git clone https://github.com/PseudoxPC/EmoNet.git
- cd EmoNet

# Create virtual environment (optional but recommended)
- python -m venv venv
- source venv/bin/activate   # For Linux/Mac
- venv\Scripts\activate      # For Windows

# Install dependencies
pip install -r requirements.txt

## ▶️ Run the App
streamlit run app.py

Now, open your browser at localhost and start detecting emotions! 🎉

## 🧠 Model Architecture

Our CNN model is designed with:

- 4️⃣ Convolutional Layers

- 🔽 MaxPooling layers for feature reduction

- ❌ Dropout layers for regularization

- 🔗 Fully Connected Dense layers

- 🔑 Output layer with Softmax activation (7 classes)

# 📊 Training Performance:

- Training Accuracy: ~75%

- Validation Accuracy: ~72%

## 🎥 Demo
- # 📷 Live Detection
Using your webcam, EmoNet detects emotions in real-time:
- # 🖼 Image Upload
Upload an image and EmoNet will classify the detected face:

## 👨‍💻 Contributors

# Priyanshu Chouhan
# Krishna Gupta



## 📜 License

This project is licensed under the MIT License – feel free to use, modify, and distribute.

<p align="center"> 🚀 Built with ❤️ using Deep Learning & Computer Vision </p> 
