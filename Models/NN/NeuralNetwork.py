from netaddr.strategy.ipv6 import num_words

from Layer import Layer
from Math.LinearAlg import Vector, dot
from gradient_descent import gradient_step
from typing import List
import random
import tqdm


class NeuralNetwork:
    def __init__(self):
        self.layers: List[Layer] = []

    def __repr__(self):
        if self.layers:
            return f"NeuralNetwork(\n" + ",\n".join("\t" + layer.__repr__() for layer in self.layers) + "\n)"
        return "NeuralNetwork(layers=[])"

    def __getitem__(self, input: Vector):
        return self.feed_forward(input)

    def feed_forward(self, input: Vector):
        outputs: List[Vector] = []

        for layer in self.layers:
            input_with_bias = input + [1] # bias de 1 como padrão
            output = [neuron.output(input_with_bias)
                      for neuron in layer.neurons]
            outputs.append(output)

            input = output

        return outputs

    def create(self, network_scheme: List[Vector]):
        self.layers = []
        for layer in network_scheme:
            l = Layer()
            l.add_neurons(layer)
            self.layers.append(l)

    def sqerror_gradients(self, input_vector: Vector, target_vector: Vector) -> List[List[Vector]]:

        # get current result
        *hidden_outputs, outputs = self.feed_forward(input_vector)

        output_deltas = [output * (1 - output) * (output - target)
                         for output, target in zip(outputs, target_vector)]

        output_grads = [[output_deltas[i] * hidden_output
                         for hidden_output in hidden_outputs[0] + [1]]
                        for i, output_neuron in enumerate(self.layers[-1].neurons)]

        hidden_deltas = [hidden_output * (1 - hidden_output) *
                        dot(output_deltas, *[n.weights for n in self.layers[-1].neurons])
                        for hidden_output in hidden_outputs[0]]

        hidden_grads = [[hidden_deltas[i] * input for input in input_vector + [1]]
                        for i, hidden_neuron in enumerate(self.layers[0].neurons)]

        return [hidden_grads, output_grads]

    def train(self, epochs: int, learning_rate: float, dataset: List[List]):
        xs, ys = dataset
        for epoch in tqdm.trange(epochs, desc="Training Started!"):
            print(f"Epoch: {epoch}")
            for x, y in zip(xs, ys):
                gradients = self.sqerror_gradients(x, y)

                network_schema = [[gradient_step(neuron.weights, grad, -learning_rate)
                                  for neuron, grad in zip(layer.neurons, layer_grad)]
                                  for layer, layer_grad in zip(self.layers, gradients)]

                self.create(network_schema)



if __name__ == '__main__':
    xor_nn = NeuralNetwork()
    xor_schematics = [
        [ # hidden layers
            [20., 20, -30],
            [20., 20, -10],
        ],
        [[-60., 60, -30]] # output layer
    ]
    xor_nn.create(xor_schematics)
    print(xor_nn)

    print(xor_nn.feed_forward([0, 0])[-1][0])
    print(xor_nn.feed_forward([1, 0])[-1][0])
    print(xor_nn.feed_forward([0, 1])[-1][0])
    print(xor_nn.feed_forward([1, 1])[-1][0])

    # xs = [[0., 0], [0., 1], [1., 0], [1., 1]]
    # ys = [[0.], [1.], [1.], [0.]]
    #
    # rand = random.Random()
    #
    # network_scheme = [
    #     [[rand.random() for _ in range(2 + 1)],
    #      [rand.random() for _ in range(2 + 1)],],
    #     [[rand.random() for _ in range(2 + 1)]]
    # ]
    #
    # network = NeuralNetwork()
    # network.create(network_scheme)
    #
    # print([n.weights for n in network.layers[-1].neurons])
    #
    # network.train(20000, 1, [xs, ys])
    #
    # print(network)
    # print(network.feed_forward([0, 0])[-1][0])
    # print(network.feed_forward([1, 0])[-1][0])
    # print(network.feed_forward([0, 1])[-1][0])
    # print(network.feed_forward([1, 1])[-1][0])