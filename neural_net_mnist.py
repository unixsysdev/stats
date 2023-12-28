from PIL import Image
import numpy as np
from tensorflow.keras.datasets import mnist
import random
import math

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def derivative_sigmoid(x):
    return x * (1 - x)

# Initialize a neural network with Xavier initialization
def initialize_network(input_size, hidden_size, output_size):
    # Xavier initialization for weights
    xavier = 1 / np.sqrt(input_size)
    network = {
        'hidden': {
            'weights': np.random.randn(input_size, hidden_size) * xavier,
            'biases': np.zeros(hidden_size),
        },
        'output': {
            'weights': np.random.randn(hidden_size, output_size) * xavier,
            'biases': np.zeros(output_size),
        }
    }
    return network

# Forward pass through the network
def forward_pass(network, inputs):
    hidden_activation = np.dot(inputs, network['hidden']['weights']) + network['hidden']['biases']
    hidden_output = sigmoid(hidden_activation)

    output_activation = np.dot(hidden_output, network['output']['weights']) + network['output']['biases']
    final_output = sigmoid(output_activation)

    return hidden_output, final_output

# Backpropagation algorithm
def backpropagate(network, inputs, expected_output, learning_rate):
    hidden_output, final_output = forward_pass(network, inputs)

    output_errors = expected_output - final_output
    output_delta = output_errors * derivative_sigmoid(final_output)

    hidden_errors = output_delta.dot(network['output']['weights'].T)
    hidden_delta = hidden_errors * derivative_sigmoid(hidden_output)

    # Update weights and biases
    network['output']['weights'] += np.outer(hidden_output, output_delta) * learning_rate
    network['output']['biases'] += np.sum(output_delta, axis=0) * learning_rate
    network['hidden']['weights'] += inputs.reshape(-1, 1).dot(hidden_delta.reshape(1, -1)) * learning_rate
    network['hidden']['biases'] += np.sum(hidden_delta, axis=0) * learning_rate

# Train the neural network
def train_network_mnist(network, train_images, train_labels, learning_rate, epochs):
    for epoch in range(epochs):
        total_errors = 0
        e = 0
        for img, label in zip(train_images, train_labels):
            if e % 5000 == 0:
                print(f"Processing Img {e}")
            e += 1
            label_one_hot = np.zeros(10)
            label_one_hot[label] = 1
            output = forward_pass(network, img)[1]
            predicted_label = np.argmax(output)
            if predicted_label != label:
                total_errors += 1
            backpropagate(network, img, label_one_hot, learning_rate)

        error_rate = total_errors / len(train_images)
        print(f"Epoch {epoch + 1}/{epochs}, Error Rate: {error_rate:.4f}")


def test_network_mnist(network, test_images, test_labels):
    total_errors = 0
    for img, label in zip(test_images, test_labels):
        label_one_hot = np.zeros(10)
        label_one_hot[label] = 1
        output = forward_pass(network, img)[1]
        predicted_label = np.argmax(output)
        if predicted_label != label:
            total_errors += 1
    error_rate = total_errors / len(test_images)
    return error_rate


if __name__ == "__main__":
    # Load and preprocess the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = np.array([np.array(Image.fromarray(img).resize((5 ,5), Image.Resampling.LANCZOS)) for img in train_images])
    test_images = np.array([np.array(Image.fromarray(img).resize((5,5), Image.Resampling.LANCZOS)) for img in test_images])
    train_images = train_images[:20000]
    train_images = train_images.reshape(train_images.shape[0], -1) / 255.0
    test_images = test_images.reshape(test_images.shape[0], -1) / 255.0

    # Define network hyperparameters
    input_size = train_images.shape[1]
    hidden_size = 32
    output_size = 10
    learning_rate = 0.1
    epochs = 10

    # Initialize and train the neural network
    network = initialize_network(input_size, hidden_size, output_size)
    train_network_mnist(network, train_images, train_labels, learning_rate, epochs)

    # Evaluate the network on the test set
    test_error_rate = test_network_mnist(network, test_images, test_labels)
    print(f"Test Error Rate: {test_error_rate:.4f}")
