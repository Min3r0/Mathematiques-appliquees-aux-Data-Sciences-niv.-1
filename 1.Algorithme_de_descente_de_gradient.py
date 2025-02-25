import numpy as np


def cost_function(x):
    return x ** 2 + 3 * x + 2


def gradient(x):
    return 2 * x + 3


def gradient_descent(lr=0.1, epochs=100, x_init=0):
    x = x_init
    history = []
    for _ in range(epochs):
        grad = gradient(x)
        x = x - lr * grad
        history.append(x)
    return x, history


optimal_x, history = gradient_descent(lr=0.1, epochs=50, x_init=5)
print(history)
print(f"Valeur optimale trouvée : {optimal_x}")
print(cost_function(history[-1]))
