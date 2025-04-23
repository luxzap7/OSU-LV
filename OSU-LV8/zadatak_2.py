import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt

# Učitavanje spremljenog modela
model = keras.models.load_model("mnist_model.h5")

# Učitavanje MNIST skupa podataka
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Skaliranje slika na raspon [0,1]
x_test_s = x_test.astype("float32") / 255

# Dodavanje dimenzije za kanale (28, 28) -> (28, 28, 1)
x_test_s = np.expand_dims(x_test_s, -1)

# Predikcija na testnom skupu
y_pred = model.predict(x_test_s)
y_pred_classes = np.argmax(y_pred, axis=1)

# Pronalazak loše klasificiranih primjera
misclassified_indices = np.where(y_pred_classes != y_test)[0]

# Prikaz nekoliko loše klasificiranih slika
plt.figure(figsize=(10, 10))
for i, idx in enumerate(misclassified_indices[:9]):  # Prikaz prvih 9 loše klasificiranih
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f"Stvarna: {y_test[idx]}, Predviđena: {y_pred_classes[idx]}")
    plt.axis('off')
plt.tight_layout()
plt.show()