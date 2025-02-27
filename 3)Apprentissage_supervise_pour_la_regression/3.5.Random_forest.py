import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).flatten() + np.random.normal(0, 0.1, 100)


n_trees = 50  # Nombre d'arbres à afficher
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_reg.fit(X, y)


X_test = np.linspace(0, 5, 100).reshape(-1, 1)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, label="Données réelles", alpha=0.6)

for i, tree in enumerate(rf_reg.estimators_[:n_trees]):  # Prendre les 5 premiers arbres
    y_tree_pred = tree.predict(X_test)
    plt.plot(X_test, y_tree_pred, linestyle="dotted", alpha=0.7, label=f"Arbre {i+1}")

y_pred = rf_reg.predict(X_test)
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Random Forest (moyenne)")

plt.title("Visualisation de plusieurs arbres dans une Random Forest")
plt.show()
