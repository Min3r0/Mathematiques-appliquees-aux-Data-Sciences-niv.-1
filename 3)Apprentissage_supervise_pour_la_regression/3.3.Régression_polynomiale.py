import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(42)
x = np.linspace(0, 10, 20).reshape(-1, 1)
y = 3 * x.flatten() + 5 + np.random.normal(0, 2, size=x.shape[0])


model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)


plt.scatter(x, y, label="Données réelles")
plt.plot(x, y_pred, color="red", label="Régression linéaire")
plt.legend()
plt.show()


print(f"Coefficient directeur (a) : {model.coef_[0]}")
print(f"Ordonnée à l'origine (b) : {model.intercept_}")
