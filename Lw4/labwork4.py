import math

# ---- helpers ----
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def sigmoid_derivative(z):
    return z * (1 - z)

# ---- Neuron ----
class Neuron:
    def __init__(self, weight=None, bias=None):
        self.weight = weight
        self.bias = bias

    def activate(self, inputs):
        z = sum([w * i for w, i in zip(self.weight, inputs)]) + self.bias
        return sigmoid(z)

# ---- Layer ----
class Layer:
    def __init__(self, neurons):
        self.neurons = neurons

    def activate(self, inputs):
        return [neuron.activate(inputs) for neuron in self.neurons]

    def forward(self, inputs):
        return self.activate(inputs)

# ---- NeuronNetwork ----
class NeuronNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs