import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)
x = np.linspace(0, 10, 20)
y = 3 * x + 5 + np.random.normal(0, 2, size=x.shape)


X = np.column_stack((x, np.ones_like(x)))


beta = np.linalg.inv(X.T @ X) @ X.T @ y
a, b = beta


y_pred = a * x + b


plt.scatter(x, y, label="Données réelles")
plt.plot(x, y_pred, color="red", label="Régression linéaire")
plt.legend()
plt.show()

print(f"Coefficient directeur (a) : {a}")
print(f"Ordonnée à l'origine (b) : {b}")
