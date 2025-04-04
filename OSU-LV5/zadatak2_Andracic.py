import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
confusion_matrix,
accuracy_score,
precision_score,
recall_score,
ConfusionMatrixDisplay,
)

labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    edgecolor = 'w',
                    label=labels[cl])

# ucitaj podatke
df = pd.read_csv("penguins.csv")

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}, inplace = True)

print(df.info())

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()

# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)


# a)
unique_classes, counts = np.unique(y_train, return_counts=True)
plt.bar(unique_classes, counts)
plt.title('Broj primjera po klasi')
plt.show()

# b,c)

# Inicijalizacija modela logističke regresije
model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=100)

# Procjena parametara modela na skupu za učenje
model.fit(X_train, y_train)

# Parametri modela (koeficijenti i intercept)
print("Koeficijenti:", model.coef_)
print("Intercept:", model.intercept_)

# d)

plot_decision_regions(X_train,y_train.ravel(),classifier=model)
plt.xlabel("Bill lenght")
plt.ylabel("Flipper lenght")
plt.show()

# e)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# 1. Klasifikacija testnog skupa
y_pred = model.predict(X_test)

# 2. Matrica zabune
cm = confusion_matrix(y_test, y_pred)
print("Matrica zabune:\n", cm)

# 3. Točnost modela
accuracy = accuracy_score(y_test, y_pred)
print("Točnost modela:", accuracy)

# 4. Izvještaj klasifikacije (četiri glavne metrike)
report = classification_report(y_test, y_pred)
print("Izvještaj klasifikacije:\n", report)

# f)
# # Primjer podataka (postojeće ulazne veličine i ciljne vrijednosti)
X = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 2]])
y = np.array([0, 0, 1, 1, 1])

# Dodavanje novih ulaznih veličina (npr. kvadrat postojećih značajki)
new_feature_1 = X[:, 0] ** 2  # Kvadrat prve značajke
new_feature_2 = X[:, 1] ** 2  # Kvadrat druge značajke

# Kombiniranje postojećih i novih značajki
X_extended = np.column_stack((X, new_feature_1, new_feature_2))

# Podjela na skup za učenje i testiranje
X_train, X_test, y_train, y_test = train_test_split(X_extended, y, test_size=0.2, random_state=42)

# Treniranje modela s proširenim skupom ulaznih veličina
model = LogisticRegression()
model.fit(X_train, y_train)

# Ispis parametara modela nakon dodavanja novih značajki
print("Intercept:", model.intercept_)
print("Koeficijenti:", model.coef_)