from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("cifar10_model.h5")
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    img = np.array(data['image']).reshape((1, 32, 32, 3))
    prediction = model.predict(img)
    return jsonify({"prediction": int(np.argmax(prediction))})

if __name__ == '__main__':
    app.run(debug=True)