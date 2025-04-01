from statistics import correlation, standard_deviation, mean
from gradient_descent import gradient_step
from Math.LinearAlg import Vector
from typing import Tuple
import random
import tqdm
import math


class SimpleLinearRegression:
    # y_i = b * x_i + a

    def __init__(self):
        self.beta = 0
        self.alpha = 0
        self.max_gradient = 1e6

    def __repr__(self):
        return f"SimpleLinearRegretion(alpha={self.alpha}, beta={self.beta})"

    def predict(self, x_i: float) -> float:
        return self.beta * x_i + self.alpha

    def error(self, x_i: float, y_i: float) -> float:
        """
        Returns a quantified ammount of the model's error being compared to x's real image (y_i)
        :param x_i: the x value
        :param y_i: the y value it should have predicted
        :return: the difference between the model's current prediction, and it's supposed y value
        """
        return self.predict(x_i) - y_i

    def sum_of_sqerrors(self, x: Vector, y: Vector) -> float:
        """
        Returns a more significant metric than just the normal error, this makes so that it avoids the complications
        that surge with -error +error equaling 0. By squaring the errors, we get rid of this difficulty
        """
        # print([self.error(x_i, y_i) for x_i, y_i in zip(x,y)])
        return sum(math.pow(self.error(x_i, y_i), 2)
                   for x_i, y_i in zip(x,y))

    def least_squares_fit(self, x: Vector, y: Vector) -> Tuple[float, float]:
        """
        Calculates alpha and beta so it minimizes errors
        :return: alpha, beta after being calculated
        """
        beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
        alpha = mean(y) - beta * mean(x)
        return alpha, beta

    def train(self, x_train, y_train, num_epochs, learning_rate):
        guess = [random.random(), random.random()]

        with tqdm.trange(num_epochs) as t:
            for _ in t:
                alpha, beta = guess
                model.alpha = alpha
                model.beta = beta

                # Derivada parcial da perda em relação a alpha
                grad_a = sum(2 * model.error(x_i, y_i)
                             for x_i, y_i in zip(x_train, y_train))

                # Derivada parcial da perda em relação a beta
                grad_b = sum(2 * model.error(x_i, y_i) * x_i
                             for x_i, y_i in zip(example_x, example_y))

                max_grad = 1e6  # Limite seguro para evitar estouros
                grad_a = max(min(grad_a, self.max_gradient), -self.max_gradient)
                grad_b = max(min(grad_b, self.max_gradient), -self.max_gradient)

                loss = model.sum_of_sqerrors(example_x, example_y)
                t.set_description(f"Loss: {loss:.5e} | dAlpha: {grad_a:.2e} | dBeta: {grad_b:.2e} | Alpha: {alpha:.5f} | Beta: {beta:.5f}")


                guess = gradient_step(guess, [grad_a, grad_b], -learning_rate)

        print('Training completed!')

if __name__ == '__main__':
    example_x = [i for i in range(-1_000, 1_001)]
    example_y = [50 * i - 25 for i in example_x]

    model = SimpleLinearRegression()
    model.train(example_x, example_y, 10_000, 0.000001)
    print(model)