# Učitavanje potrebnih biblioteka
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

# Učitavanje podataka
data = pd.read_csv('Social_Network_Ads.csv')

# Odabir ulaznih i izlaznih veličina
X = data[['Age', 'EstimatedSalary']].values
y = data['Purchased'].values

# Podjela podataka na skup za učenje i testiranje (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizacija podataka
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Funkcija za vizualizaciju granice odluke
def plot_decision_regions(X, y, classifier, resolution=0.01):
    # Postavljanje markera i mreže
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Granice odluke
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Točke podataka
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=f'Class {cl}', edgecolor='black')

# 1. SVM model s RBF kernelom
svm_rbf = SVC(kernel='rbf', C=1.0, gamma=0.1, random_state=42)
svm_rbf.fit(X_train, y_train)

# Točnost na skupu za učenje i testiranje
train_acc_rbf = svm_rbf.score(X_train, y_train)
test_acc_rbf = svm_rbf.score(X_test, y_test)

print(f"SVM (RBF kernel, C=1.0, gamma=0.1): Train acc={train_acc_rbf:.2f}, Test acc={test_acc_rbf:.2f}")

# Vizualizacija granice odluke za RBF kernel
plot_decision_regions(X_test, y_test, classifier=svm_rbf)
plt.title('SVM (RBF kernel, C=1.0, gamma=0.1) - Granica odluke')
plt.legend()
plt.show()

# 2. Promjena hiperparametara C i gamma
for C in [0.1, 1, 10]:
    for gamma in [0.01, 0.1, 1]:
        svm_rbf = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
        svm_rbf.fit(X_train, y_train)
        test_acc = svm_rbf.score(X_test, y_test)
        print(f"SVM (RBF kernel, C={C}, gamma={gamma}): Test acc={test_acc:.2f}")
        plot_decision_regions(X_test, y_test, classifier=svm_rbf)
        plt.title(f'SVM (RBF kernel, C={C}, gamma={gamma}) - Granica odluke')
        plt.legend()
        plt.show()

# 3. Promjena tipa kernela
for kernel in ['linear', 'poly', 'sigmoid']:
    svm = SVC(kernel=kernel, C=1.0, gamma=0.1, random_state=42)
    svm.fit(X_train, y_train)
    test_acc = svm.score(X_test, y_test)
    print(f"SVM ({kernel} kernel): Test acc={test_acc:.2f}")
    plot_decision_regions(X_test, y_test, classifier=svm)
    plt.title(f'SVM ({kernel} kernel) - Granica odluke')
    plt.legend()
    plt.show()