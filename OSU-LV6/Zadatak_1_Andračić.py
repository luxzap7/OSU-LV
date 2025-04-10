# Učitavanje potrebnih biblioteka
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

# 1. Izrada KNN modela (K=5)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Točnost na skupu za učenje i testiranje
train_acc_knn = knn.score(X_train, y_train)
test_acc_knn = knn.score(X_test, y_test)

# Usporedba s logističkom regresijom
log_reg = LogisticRegression(solver='lbfgs', max_iter=1000)
log_reg.fit(X_train, y_train)
train_acc_lr = log_reg.score(X_train, y_train)
test_acc_lr = log_reg.score(X_test, y_test)

print(f"KNN (K=5): Train acc={train_acc_knn:.2f}, Test acc={test_acc_knn:.2f}")
print(f"Logistička regresija: Train acc={train_acc_lr:.2f}, Test acc={test_acc_lr:.2f}")

# Vizualizacija granice odluke za K=5
plot_decision_regions(X_test, y_test, classifier=knn)
plt.title('KNN (K=5) - Granica odluke')
plt.legend()
plt.show()

# 2. Granica odluke za K=1 i K=100
knn_k1 = KNeighborsClassifier(n_neighbors=1)
knn_k1.fit(X_train, y_train)
plot_decision_regions(X_test, y_test, classifier=knn_k1)
plt.title('KNN (K=1) - Granica odluke')
plt.legend()
plt.show()

knn_k100 = KNeighborsClassifier(n_neighbors=100)
knn_k100.fit(X_train, y_train)
plot_decision_regions(X_test, y_test, classifier=knn_k100)
plt.title('KNN (K=100) - Granica odluke')
plt.legend()
plt.show()