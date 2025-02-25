import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x = np.linspace(0, 10, 20)
y = 3 * x + 5 + np.random.normal(0, 2, size=x.shape)  # y = 3x + 5 + bruit

a_values = np.linspace(0, 6, 100)
b_values = np.linspace(0, 10, 100)

best_a, best_b = 0, 0
min_error = float("inf")

for a in a_values:
    for b in b_values:
        y_pred = a * x + b
        mse = np.mean((y - y_pred) ** 2)
        if mse < min_error:
            min_error = mse
            best_a, best_b = a, b

print(f"Meilleurs paramètres trouvés : a = {best_a}, b = {best_b}")

plt.scatter(x, y, label="Données réelles")
plt.plot(x, best_a * x + best_b, color="red", label="Régression par force brute")
plt.legend()
plt.show()
