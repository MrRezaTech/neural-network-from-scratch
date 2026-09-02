# Neural Network from Scratch (NumPy)

A modular Deep Neural Network built completely **from scratch** using Python and **NumPy**. This project demonstrates the fundamental mathematics of deep learning without relying on high-level frameworks like PyTorch or TensorFlow.

---

## 🔑 Key Features
* **Modular Architecture**: Supports an arbitrary number of hidden layers and nodes.
* **Activations**: Implemented **ReLU** for hidden layers and **Softmax** for multi-class classification.
* **Initialization**: Uses **He (Kaiming) Initialization** optimized for ReLU activation.
* **Optimizer**: Built-in **RMSprop** optimizer for adaptive learning rates and fast convergence.
* **Numerical Stability**: Handles overflow in Softmax by normalizing logits ($Z - \max(Z)$) and clips probabilities in Cross-Entropy Loss.
* **Visualization**: Built-in methods to plot training loss curves over epochs.

---

## 📐 Mathematical Formulation

### 1. Forward Pass
For each layer $l$:
$$Z^{(l)} = A^{(l-1)} W^{(l)} + b^{(l)}$$
$$A^{(l)} = \sigma(Z^{(l)})$$

### 2. Loss Function (Categorical Cross-Entropy)
$$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_{i,k} \log(\hat{y}_{i,k})$$

### 3. Backpropagation & Optimization
* **Output Layer Error**:
  $$\delta^{(L)} = A^{(L)} - Y$$
* **Hidden Layers Error**:
  $$\delta^{(l)} = (\delta^{(l+1)} (W^{(l+1)})^T) \odot \sigma'(Z^{(l)})$$
* **Gradients**:
  $$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{1}{m} (A^{(l-1)})^T \delta^{(l)}, \quad \frac{\partial \mathcal{L}}{\partial b^{(l)}} = \frac{1}{m} \sum_{i=1}^{m} \delta_i^{(l)}$$

---

## 🚀 Quick Start

### Installation
Ensure you have Python installed, then install the required dependencies:
```bash
pip install numpy matplotlib
from Neural_Network import NeuralNetwork
import numpy as np

# Sample Data
X_train = np.array([[1,3,4,5,6,4,2], [3,2,3,4,5,6,2], [1,2,3,2,4,3,5]])
y_train = np.array([[0,0,0,0,0,1], [0,0,1,0,0,0], [0,1,0,0,0,0]])

# Initialize Model: 3 hidden layers + 1 output layer (6 classes)
model = NeuralNetwork(
    layer_sizes=[8, 8, 8, 6], 
    activations=["relu", "relu", "relu", "softmax"], 
    input_dim=X_train.shape[1]
)

# Train the Model
history = model.train(X_train, y_train, epochs=500, lr=0.01)

# Plot Training Loss
model.chart()

# Predictions
predictions = model.predict(X_train)
print("Predictions:\n", np.round(predictions, 2))```