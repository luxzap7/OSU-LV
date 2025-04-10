# Učitavanje potrebnih biblioteka
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

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

# Definicija SVM modela
svm = SVC(kernel='rbf', random_state=42)

# Definicija raspona hiperparametara C i gamma za pretragu
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10]
}

# Unakrsna validacija za odabir optimalnih hiperparametara
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Prikaz optimalnih vrijednosti hiperparametara i pripadajuće točnosti
optimal_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Optimalne vrijednosti hiperparametara: C={optimal_params['C']}, gamma={optimal_params['gamma']}")
print(f"Najbolja točnost tijekom unakrsne validacije: {best_score:.2f}")

# Evaluacija modela s optimalnim hiperparametrima na testnom skupu
svm_optimal = SVC(kernel='rbf', C=optimal_params['C'], gamma=optimal_params['gamma'], random_state=42)
svm_optimal.fit(X_train, y_train)
test_accuracy = svm_optimal.score(X_test, y_test)

print(f"Točnost na testnom skupu s optimalnim hiperparametrima: {test_accuracy:.2f}")