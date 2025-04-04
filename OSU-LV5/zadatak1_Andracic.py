import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


X, y = make_classification(
n_samples=200,
n_features=2,
n_redundant=0,
n_informative=2,
random_state=213, # 213
n_clusters_per_class=1,
class_sep=1,
)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
## template ⬆️


# a) Prikaz podataka za učenje u x1 x2 ravnini
plt.figure() # Postavljanje figure
plt.scatter(
X_train[:, 0], X_train[:, 1], c=y_train, label="Training"
) # Prikaz podataka za učenje

"""
plt.scatter(
X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", label="Testing", marker="x"
) # Prikaz podataka za testiranje
"""
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("a) Podaci za učenje")
plt.legend()


# b) model logističke regresije pomocu scikit-learn
from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train) # Učenje modela


# c) pronaci u atributima izgradenog modela parametre modela.
# prikazati granicu odluke naucenog modela u x1 x2 ravnini
# zajedno s podacima za učenje
# parametri modela
theta0 = logistic_model.intercept_[0] # Intercept (bias) modela
theta1, theta2 = logistic_model.coef_[0] # Koeficijenti modela
# granica odluke
x1_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
x2_vals = -(theta0 + theta1 * x1_vals) / theta2

plt.figure(figsize=(8, 6)) # Postavljanje figure veličine 8x6 inch
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", label="Training")
plt.plot(x1_vals, x2_vals, color="black", label="granica odluke")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("C) Granica odluke sa trening podacima")
plt.legend()

# d) klasifikacija skupa podataka za testiranje pomocu
# izgrađenog modela log regresije. izralunati i prikazati
# matricu zabune na testnim podacima
# točnost, preciznost i odziv na testnom skupu
from sklearn.metrics import (
confusion_matrix,
accuracy_score,
precision_score,
recall_score,
ConfusionMatrixDisplay,
)

y_pred = logistic_model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
display = ConfusionMatrixDisplay(
confusion_matrix=conf_matrix, display_labels=logistic_model.classes_
)
display.plot(cmap="Blues")
plt.title("d) Matrica zabune")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f"Točnost: {accuracy:.2f}")
print(f"Preciznost: {precision:.2f}")
print(f"Odziv: {recall:.2f}")

# e) prikaz testnog skupa u x1 x2 ravnini. zeleno dobro klasificirani
# crno lose klasificirani uzorci
correct = y_test == y_pred # usporedba stvarnih i predikcija
incorrect = ~correct # tilda negira bool vrijednost

plt.figure(figsize=(8, 6))
plt.scatter(
X_test[correct, 0],
X_test[correct, 1],
color="green",
label="Dobro klasificirani",
alpha=0.7,
)
plt.scatter(
X_test[incorrect, 0],
X_test[incorrect, 1],
color="black",
label="pogresno kvalificirani",
alpha=0.7,
)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("e) Test Data Classification")
plt.legend()
plt.show()