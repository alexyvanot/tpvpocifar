import streamlit as st
from PIL import Image
import numpy as np
import requests

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

st.title("Démo modèle CIFAR-10")

uploaded_file = st.file_uploader("Uploader une image 32x32", type=["png", "jpg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((32,32))
    st.image(image, caption="Image à prédire")
    img = np.array(image).astype('float32') / 255.0
    response = requests.post("http://localhost:5000/predict", json={'image': img.tolist()})
    prediction = response.json()['prediction']
    st.write(f"Prédiction : {classes[prediction]}")