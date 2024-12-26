import numpy as np
import matplotlib.pyplot as plt


# 1. Загрузка данных из файла
def load_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    X = data[:, :-1]  # Признаки: вибрация и неравномерность вращения
    y = data[:, -1]  # Целевая переменная: 0 или 1
    return X, y


# 2. Нормализация данных (собственная реализация mean и std)
def mean_custom(X):
    return np.sum(X, axis=0) / X.shape[0]


def std_custom(X, mean):
    return np.sqrt(np.sum((X - mean) ** 2, axis=0) / X.shape[0])


def normalize_features(X):
    mean = mean_custom(X)
    std = std_custom(X, mean)
    X_norm = (X - mean) / std
    return X_norm, mean, std


# 3. Добавление нелинейных признаков (полиномы до 3-й степени)
def add_nonlinear_features(X):
    x1 = X[:, 0]
    x2 = X[:, 1]
    new_features = np.c_[X, x1 ** 2, x2 ** 2, x1 * x2, x1 ** 3, x2 ** 3, x1 ** 2 * x2, x1 * x2 ** 2]
    return new_features


# 4. Функция сигмоиды с обработкой переполнения
def sigmoid(z):
    z = np.clip(z, -500, 500)  # Ограничиваем значения для избежания переполнения
    return 1 / (1 + np.exp(-z))


# 5. Функция стоимости (логистическая) с защитой от log(0)
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    epsilon = 1e-8  # Добавляем малое значение для избежания log(0)
    cost = -(1 / m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
    return cost


# 6. Градиентный спуск для оптимизации весов
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    cost_history = []

    for _ in range(num_iters):
        h = sigmoid(X.dot(theta))
        gradient = (1 / m) * X.T.dot(h - y)
        theta -= alpha * gradient
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history


# 7. Предсказание на основе входных данных
def predict(theta, input_features, mean, std):
    input_normalized = (input_features - mean) / std
    input_with_features = add_nonlinear_features(input_normalized.reshape(1, -1))
    input_with_bias = np.insert(input_with_features, 0, 1)  # Добавляем bias
    probability = sigmoid(np.dot(input_with_bias, theta))
    return probability


# 8. Основная функция
if __name__ == "__main__":
    # Устанавливаем безопасный бэкенд matplotlib
    import matplotlib

    matplotlib.use('TkAgg')

    file_path = 'ex2data1.txt'  # Путь к файлу
    X, y = load_data(file_path)

    # Нормализация данных
    X_normalized, mean, std = normalize_features(X)

    # Добавляем нелинейные признаки
    X_nonlinear = add_nonlinear_features(X_normalized)

    # Добавляем единичный столбец для theta0 (сдвиг)
    X_with_bias = np.c_[np.ones(X_nonlinear.shape[0]), X_nonlinear]

    # Инициализация параметров
    theta = np.zeros(X_with_bias.shape[1])
    alpha = 0.01  # Скорость обучения
    num_iters = 30000  # Количество итераций

    # Оптимизация параметров с помощью градиентного спуска
    theta_optimized, cost_history = gradient_descent(X_with_bias, y, theta, alpha, num_iters)

    # Вывод результатов
    print("Оптимизированные параметры (theta):", theta_optimized)
    print("Финальная стоимость:", cost_history[-1])

    # Визуализация функции стоимости
    plt.plot(range(num_iters), cost_history)
    plt.xlabel('Итерации')
    plt.ylabel('Функция стоимости')
    plt.title('Сходимость функции стоимости')
    plt.show()

    # Визуализация данных и границы раздела
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', label='Класс 0')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label='Класс 1')
    plt.xlabel('Вибрация')
    plt.ylabel('Неравномерность вращения')
    plt.legend()
    plt.title('Данные и граница раздела')

    # Построение границы раздела с денормализацией
    u = np.linspace(-2, 2, 100)  # Нормализованные координаты
    v = np.linspace(-2, 2, 100)
    z = np.zeros((len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            features = np.array([u[i], v[j]])
            features = add_nonlinear_features(features.reshape(1, -1))
            features = np.insert(features, 0, 1)  # Добавляем bias
            z[i, j] = np.dot(features, theta_optimized)

    # Денормализация для отображения на графике
    u_denorm = u * std[0] + mean[0]
    v_denorm = v * std[1] + mean[1]
    plt.contour(u_denorm, v_denorm, z.T, levels=[0], colors='green')
    plt.show()

    # Ввод пользовательских данных для предсказания
    while True:
        try:
            user_input = input(
                "Введите признаки (вибрация, неравномерность вращения) через запятую или 'exit' для выхода: ")
            if user_input.lower() == 'exit':
                break
            input_features = np.array([float(x) for x in user_input.split(',')])
            prediction = predict(theta_optimized, input_features, mean, std)
            print(f"Вероятность неисправности двигателя: {prediction:.4f}")
        except Exception as e:
            print("Ошибка ввода данных. Попробуйте снова.", e)
