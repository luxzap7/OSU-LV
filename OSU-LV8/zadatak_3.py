import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
from PIL import Image

# Učitavanje spremljenog modela
model = keras.models.load_model("mnist_model.h5")

# Učitavanje slike sa diska
image_path = "test.png"
img = Image.open(image_path).convert('L')  # Pretvori u grayscale
img = img.resize((28, 28))  # Promijeni veličinu na 28x28 piksela
img_array = np.array(img)

# Prikaz originalne slike
plt.imshow(img_array, cmap='gray')
plt.title("Učitana slika")
plt.axis('off')
plt.show()

# Prilagodba slike za mrežu
img_array = img_array.astype("float32") / 255  # Skaliranje na raspon [0,1]
img_array = np.expand_dims(img_array, axis=-1)  # Dodavanje dimenzije za kanale
img_array = np.expand_dims(img_array, axis=0)  # Dodavanje dimenzije za batch

# Klasifikacija slike
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

# Ispis rezultata
print(f"Predviđena oznaka za sliku: {predicted_class}")