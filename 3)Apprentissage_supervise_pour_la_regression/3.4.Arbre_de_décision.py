import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression

np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).flatten() + np.random.normal(0, 0.1, 80)

tree_reg = DecisionTreeRegressor(max_depth=3)
tree_reg.fit(X, y)

X_test = np.linspace(0, 5, 100).reshape(-1, 1)
y_pred = tree_reg.predict(X_test)

plt.scatter(X, y, label="Données réelles")
plt.plot(X_test, y_pred, color="red", label="Prédiction de l'arbre de décision")
plt.legend()
plt.show()
