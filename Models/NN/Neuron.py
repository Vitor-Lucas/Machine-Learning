from Math.LinearAlg import Vector, dot
import math

class Neuron:
    def __init__(self, weights=None):
        if weights is None:
            weights = []
        self.weights:Vector = weights

    def output(self, inputs: Vector) -> float:
        weighted_sum = dot(self.weights, inputs)
        print(f"Neuron Weights: {self.weights}")
        print(f"Inputs: {inputs}")
        print(f"Dot Product: {weighted_sum}")
        print(f"Sigmoid Output: {sigmoid(weighted_sum)}\n")
        return sigmoid(weighted_sum)



def sigmoid(t: float) -> float:
    return 1 / (1 + math.exp(-t))
