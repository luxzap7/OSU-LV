import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# train i test podaci
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# prikaz karakteristika train i test podataka
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# Prikaz nekoliko slika iz train skupa
for i in range(5):
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Oznaka: {y_train[i]}")
    plt.show()

# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")

# pretvori labele
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)

# Kreiranje modela pomocu keras.Sequential()
model = keras.Sequential([
    layers.Flatten(input_shape=input_shape),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Prikaz strukture mreže
model.summary()

# Definiranje karakteristika procesa učenja
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Provođenje učenja mreže
history = model.fit(x_train_s, y_train_s, epochs=10, batch_size=32, validation_split=0.2)

# Evaluacija mreže na testnom skupu
test_loss, test_acc = model.evaluate(x_test_s, y_test_s, verbose=2)
print(f"Testna točnost: {test_acc}")

# Predikcija na testnom skupu
y_pred = model.predict(x_test_s)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test_s, axis=1)

# Prikaz matrice zabune
conf_matrix = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=range(num_classes))
disp.plot(cmap='viridis')
plt.show()

# Spremanje modela
model.save("mnist_model.h5")
print("Model je spremljen na disk.")

