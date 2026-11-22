import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

# архитектура нейросети по шаблону mnist
class NeuralNetwork:
    def __init__(self, input_nodes=5, hidden_nodes=16, output_nodes=5, learning_rate=0.15):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # случайная инициализация весов
        self.weights_ih = np.random.normal(0.0, pow(self.input_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.weights_ho = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.output_nodes, self.hidden_nodes))

        # функция активации сигмоида
        self.activation = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    # прямое распространение
    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.weights_ih, inputs)
        hidden_outputs = self.activation(hidden_inputs)
        final_inputs = np.dot(self.weights_ho, hidden_outputs)
        final_outputs = self.activation(final_inputs)
        return final_outputs

    # обратное распространение ошибки и обучение
    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.weights_ih, inputs)
        hidden_outputs = self.activation(hidden_inputs)

        final_inputs = np.dot(self.weights_ho, hidden_outputs)
        final_outputs = self.activation(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.weights_ho.T, output_errors)

        self.weights_ho += self.learning_rate * np.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)),
            hidden_outputs.T
        )
        self.weights_ih += self.learning_rate * np.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
            inputs.T
        )

# загрузка и нормализация данных
df = pd.read_csv("clustered_dataset.csv")
feature_cols = [c for c in df.columns if c.startswith("feature_")]

X = df[feature_cols].values
y = df["cluster"].values

# масштабирование признаков в диапазон [0.01, 1.0] как в mnist
x_min = X.min(axis=0)
x_max = X.max(axis=0)
X_norm = 0.01 + 0.99 * (X - x_min) / (x_max - x_min)

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)

# обучение нейросети
nn = NeuralNetwork(input_nodes=5, hidden_nodes=16, output_nodes=5, learning_rate=0.15)
epochs = 5

print("Старт обучения нейросети...")
for epoch in range(epochs):
    for i in range(len(X_train)):
        targets = np.zeros(5) + 0.01
        targets[y_train[i]] = 0.99
        nn.train(X_train[i], targets)
    print(f"Эпоха {epoch + 1}/{epochs} завершена")

# оценка точности на тесте
correct = 0
for i in range(len(X_test)):
    out = nn.query(X_test[i])
    pred = np.argmax(out)
    if pred == y_test[i]:
        correct += 1

accuracy = (correct / len(X_test)) * 100
print(f"Точность нейросети на тесте: {accuracy:.2f}%")

# сохранение весов модели
weights_data = {
    "weights_ih": nn.weights_ih,
    "weights_ho": nn.weights_ho,
    "x_min": x_min,
    "x_max": x_max
}

with open("nn_weights.pkl", "wb") as f:
    pickle.dump(weights_data, f)

print("Веса нейросети сохранены в nn_weights.pkl")
