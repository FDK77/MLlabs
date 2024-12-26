import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import matplotlib
matplotlib.use('TkAgg')

# Функция сигмоиды
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Производные сигмоиды и гиперболического тангенса
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_derivative(x):
    return 1 - tanh(x)**2  # Используем нашу реализацию tanh

# Явные формулы гиперболических функций
def sinh(x):
    return (np.exp(x) - np.exp(-x)) / 2

def cosh(x):
    return (np.exp(x) + np.exp(-x)) / 2

def tanh(x):
    return sinh(x) / cosh(x)

def sech(x):
    return 1 / cosh(x)

def coth(x):
    return cosh(x) / sinh(x)

def csch(x):
    return 1 / sinh(x)

# Точки для отображения
points = [0, 3, -3, 8, -8, 15, -15]
sigmoid_values = [sigmoid(x) for x in points]
sigmoid_deriv_values = [sigmoid_derivative(x) for x in points]
tanh_deriv_values = [tanh_derivative(x) for x in points]

# Расширенный диапазон x для графиков
x_vals = np.linspace(-20, 20, 2000)  # Расширяем диапазон для охвата точек

# Построение графика сигмоиды
plt.figure(figsize=(10, 6))
plt.plot(x_vals, [sigmoid(x) for x in x_vals], label="Sigmoid", color='b')
plt.scatter(points, sigmoid_values, color='red', zorder=5, label="Points")  # Отмечаем точки
plt.title("Sigmoid Function")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.legend()
plt.show()

# Графики гиперболических функций
x_left = np.linspace(-20, -0.1, 1000)  # Левая часть coth(x) и csch(x)
x_right = np.linspace(0.1, 20, 1000)   # Правая часть coth(x) и csch(x)

functions = {
    "sinh(x)": [(x_vals, [sinh(x) for x in x_vals])],
    "cosh(x)": [(x_vals, [cosh(x) for x in x_vals])],
    "tanh(x)": [(x_vals, [tanh(x) for x in x_vals])],
    "sech(x)": [(x_vals, [sech(x) for x in x_vals])],
    "coth(x)": [(x_left, [coth(x) for x in x_left]), (x_right, [coth(x) for x in x_right])],
    "csch(x)": [(x_left, [csch(x) for x in x_left]), (x_right, [csch(x) for x in x_right])],
}

# Цвета для каждой функции
colors = {
    "sinh(x)": "tab:blue",
    "cosh(x)": "tab:orange",
    "tanh(x)": "tab:green",
    "sech(x)": "tab:red",
    "coth(x)": "tab:purple",
    "csch(x)": "tab:brown",
}

fig, ax = plt.subplots(figsize=(12, 8))
lines = {}

# Построение графиков
for name, segments in functions.items():
    lines[name] = []
    for x_data, y_data in segments:
        line, = ax.plot(x_data, y_data, label=name, linewidth=1.5, color=colors[name])
        lines[name].append(line)

plt.title("Hyperbolic Functions")
plt.xlabel("x")
plt.ylabel("y")
plt.ylim(-10, 10)
plt.legend()
plt.grid()

# CheckButtons для включения/выключения функций
rax = plt.axes([0.87, 0.15, 0.12, 0.5], facecolor='lightgoldenrodyellow')
check = CheckButtons(rax, list(functions.keys()), [True] * len(functions))

def toggle_visibility(label):
    for line in lines[label]:
        line.set_visible(not line.get_visible())
    plt.draw()

check.on_clicked(toggle_visibility)
plt.show()

# График производной сигмоиды
plt.figure(figsize=(10, 6))
plt.plot(x_vals, [sigmoid_derivative(x) for x in x_vals], label="Sigmoid Derivative", color='r')
plt.scatter(points, sigmoid_deriv_values, color='blue', zorder=5, label="Points")  # Отмечаем точки
plt.title("Derivative of Sigmoid Function")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.legend()
plt.show()

# График производной гиперболического тангенса
plt.figure(figsize=(10, 6))
plt.plot(x_vals, [tanh_derivative(x) for x in x_vals], label="Tanh Derivative", color='g')
plt.scatter(points, tanh_deriv_values, color='purple', zorder=5, label="Points")  # Отмечаем точки
plt.title("Derivative of Tanh Function")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.legend()
plt.show()
