import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Partie 1 : Chargement et préparation des données
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# Affichage de la répartition des classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
counts = np.bincount(y_train.flatten())
plt.figure(figsize=(10,6))
plt.bar(classes, counts)
plt.title("Répartition des classes dans CIFAR-10")
plt.xticks(rotation=45)
plt.show()

# Partie 2 : Modélisation
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entraînement du modèle
early_stop = EarlyStopping(patience=3, restore_best_weights=True)
model.fit(x_train, y_train_cat, epochs=10, validation_split=0.2, callbacks=[early_stop], batch_size=64)

# Partie 3 : Analyse des performances
loss, accuracy = model.evaluate(x_test, y_test_cat)
print("Test accuracy:", accuracy)

# Partie 4 : Sauvegarde du modèle
model.save("cifar10_model.h5")