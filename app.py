# app.py

import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from datetime import datetime
import pandas as pd
import os

# Chargement du modèle entraîné
model = tf.keras.models.load_model("emotion_model.h5")

# Labels d’émotions (doivent correspondre à l'ordre de ton modèle)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Enregistrement des prédictions
def log_prediction(name, emotion):
    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    log_path = "detection_log.csv"
    entry = pd.DataFrame([[name, emotion, time_str]], columns=["Nom", "Émotion", "Horodatage"])
    if os.path.exists(log_path):
        entry.to_csv(log_path, mode='a', header=False, index=False)
    else:
        entry.to_csv(log_path, index=False)

# Prédire une émotion à partir d’une image
def predict_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face_resized = cv2.resize(gray, (48, 48)).reshape(1, 48, 48, 1) / 255.0
    prediction = model.predict(face_resized)
    emotion = emotion_labels[np.argmax(prediction)]
    return emotion

# Interface Streamlit
st.title("🧠 Détection des Émotions à partir d'une Image")
name = st.text_input("Entrez votre nom (obligatoire)", value="Stagiaire")

uploaded_file = st.file_uploader("Téléversez une image faciale (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="Image reçue", use_column_width=True)

    if st.button("Prédire l'émotion"):
        if name.strip() == "":
            st.warning("Veuillez entrer votre nom avant de prédire.")
        else:
            emotion = predict_emotion(img_rgb)
            st.success(f"Émotion détectée : **{emotion}**")
            log_prediction(name, emotion)
