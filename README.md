# TPVPOCIFAR - Computer Vision Pipeline with CIFAR-10

This project demonstrates a complete computer vision pipeline using the CIFAR-10 dataset. It covers:

- Data loading and exploration
- Dataset preprocessing
- Model building using Convolutional Neural Networks (CNN)
- Model training and evaluation
- Visualization of performance
- Model saving for inference
- Live prediction via Flask API and Streamlit interface

---

## 🚀 How to Run

### 1. Setup Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Train the model
```bash
python training_notebook.py
```

### 3. Start the Flask API
```bash
python flask_api.py
```

### 4. Run the Streamlit Interface
In another terminal:
```bash
streamlit run streamlit_interface.py
```

---

## 🧪 How it Works

- The model is trained on 10 categories of 32x32 RGB images.
- A CNN with 2 convolution layers + dense layers performs classification.
- A user can upload their own image (must be 32x32x3) to get a prediction.
- The Streamlit app sends the image to the Flask API which returns the predicted class.

---

## 🔍 Dataset Info

**CIFAR-10** contains 60,000 32x32 color images in 10 classes:
- airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

More info: https://www.cs.toronto.edu/~kriz/cifar.html

---

## 🛠️ Requirements

- Python 3.9+
- TensorFlow
- NumPy, Matplotlib
- Flask
- Streamlit
- Pillow
- Requests

Install with:
```bash
pip install -r requirements.txt
```
