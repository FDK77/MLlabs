import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
# Уравнение границы: x1^2 + x1*x2 + 2*x2 + 0.5 = 0
def decision_boundary(x1, x2):
    return x1**2 + x1 * x2 + 2 * x2 + 0.5

# Генерация сетки значений x1 и x2
x1 = np.linspace(-3, 3, 400)
x2 = np.linspace(-3, 3, 400)
x1, x2 = np.meshgrid(x1, x2)

# Подсчёт значений для границы
z = decision_boundary(x1, x2)

# Построение графика
plt.figure(figsize=(8, 6))
plt.contourf(x1, x2, z, levels=[-10, 0, 10], colors=('lightblue', 'lightgreen'))
plt.contour(x1, x2, z, levels=[0], colors='red', linewidths=2)
plt.title("Граница решения для модели")
plt.xlabel("x1")
plt.ylabel("x2")
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()