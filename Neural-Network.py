import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    """
    A modular Deep Neural Network implemented from scratch using NumPy.
    Supports arbitrary depth, ReLU/Softmax activations, and RMSprop optimization.
    """
    def __init__(self, layer_sizes: list, activations: list, input_dim: int):
        self.beta = 0.9
        self.num_layers = len(layer_sizes)
        self.weights = {}
        self.biases = {}
        self.u_w = {}
        self.u_b = {}
        self.activations = activations

        current_dim = input_dim
        for i, (nodes, act) in enumerate(zip(layer_sizes, activations), start=1):
            # He Initialization for ReLU layers
            self.weights[f"layer{i}"] = np.random.normal(0.0, np.sqrt(2 / current_dim), (current_dim, nodes))
            self.biases[f"layer{i}"] = np.zeros((1, nodes))
            
            self.u_w[f"layer{i}"] = np.zeros_like(self.weights[f"layer{i}"])
            self.u_b[f"layer{i}"] = np.zeros_like(self.biases[f"layer{i}"])
            current_dim = nodes

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    @staticmethod
    def softmax(x):
        x_max = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    @staticmethod
    def compute_loss(y_true, y_pred):
        # Categorical Cross-Entropy Loss with epsilon for numerical stability
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    def _rmsprop(self, param, grad, u, lr):
        u = self.beta * u + (1 - self.beta) * (grad ** 2)
        param -= lr * grad / (np.sqrt(u) + 1e-8)
        return param, u

    def forward(self, X):
        z_values = {}
        a_values = {"layer0": X}
        current_a = X

        for i in range(1, self.num_layers + 1):
            key = f"layer{i}"
            z = np.dot(current_a, self.weights[key]) + self.biases[key]
            
            if self.activations[i - 1].lower() == "relu":
                current_a = self.relu(z)
            elif self.activations[i - 1].lower() == "softmax":
                current_a = self.softmax(z)
                
            z_values[key] = z
            a_values[key] = current_a

        return z_values, a_values

    def train(self, X, y, epochs=1000, lr=0.001):
        m = X.shape[0]
        self.loss_history = []
        self.epochs=epochs

        for epoch in range(epochs):
            # 1. Forward Pass
            z_values, a_values = self.forward(X)
            
            # Compute Loss
            loss = self.compute_loss(y, a_values[f"layer{self.num_layers}"])
            self.loss_history.append(loss)

            # 2. Backward Pass (Fully Dynamic)
            grads_w = {}
            grads_b = {}

            # Output layer error (Softmax + Cross-Entropy derivative)
            last_key = f"layer{self.num_layers}"
            delta = (a_values[last_key] - y) / m

            grads_w[last_key] = np.dot(a_values[f"layer{self.num_layers - 1}"].T, delta)
            grads_b[last_key] = np.sum(delta, axis=0, keepdims=True)

            # Hidden layers error propagation
            for i in range(self.num_layers - 1, 0, -1):
                curr_key = f"layer{i}"
                next_key = f"layer{i + 1}"
                prev_a_key = f"layer{i - 1}"

                delta = np.dot(delta, self.weights[next_key].T) * self.relu_derivative(z_values[curr_key])
                grads_w[curr_key] = np.dot(a_values[prev_a_key].T, delta)
                grads_b[curr_key] = np.sum(delta, axis=0, keepdims=True)

            # 3. Update Weights using RMSprop
            for i in range(1, self.num_layers + 1):
                key = f"layer{i}"
                self.weights[key], self.u_w[key] = self._rmsprop(self.weights[key], grads_w[key], self.u_w[key], lr)
                self.biases[key], self.u_b[key] = self._rmsprop(self.biases[key], grads_b[key], self.u_b[key], lr)

        return self.loss_history

    def predict(self, X):
        _, a_values = self.forward(X)
        return a_values[f"layer{self.num_layers}"]

    def chart(self):

      plt.figure(figsize=(8, 5))
      plt.plot(range(1, self.epochs + 1), self.loss_history, color='blue', linewidth=2)
      plt.title("Training Loss Over Epochs")
      plt.xlabel("Epoch")
      plt.ylabel("Loss")
      plt.grid(True)
      plt.show()

# --- Example Usage ---
X_train = np.array([[1,3,4,5,6,4,2], [3,2,3,4,5,6,2], [1,2,3,2,4,3,5]])
y_train = np.array([[0,0,0,0,0,1], [0,0,1,0,0,0], [0,1,0,0,0,0]])

model = NeuralNetwork(
    layer_sizes=[8, 8, 8, 6], 
    activations=["relu", "relu", "relu", "softmax"], 
    input_dim=X_train.shape[1]
)

history = model.train(X_train, y_train, epochs=500, lr=0.01)
print("Predictions:\n", np.round(model.predict(X_train), 2))