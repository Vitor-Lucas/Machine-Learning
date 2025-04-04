from Neuron import Neuron
from typing import List
from Math.LinearAlg import Vector

class Layer:
    def __init__(self):
        self.neurons: List[Neuron] = []

    def __repr__(self):
        return f"Layer(neuron_count={len(self.neurons)})"

    def add_neurons(self, neurons_weights: Vector):
        for weights in neurons_weights:
            self.neurons.append(Neuron(weights))



