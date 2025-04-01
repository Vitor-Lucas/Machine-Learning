import random

from Math.LinearAlg import Vector, distance, add, scalar_multiply
from typing import Callable, List

def difference_quotient(f: Callable[[float], float],
                        x: float,
                        h: float) -> float:
    """
    Calculates the y value of the derivative of f
    :param f: the function
    :param x: the given x coordinate
    :param h: the value that approaches 0
    :return: the inclination of that given x coordinate
    """
    return (f(x + h) - f(x)) / h

def partial_difference_quotient(f: Callable[[Vector], float],
                                v: Vector,
                                i: int,
                                h: float) -> float:
    w = [v_j + (h if j == i else 0)
         for j, v_j in enumerate(v)]
    return (f(w) - f(v)) / h

def estimate_gradient(f: Callable[[Vector], float],
                      v: Vector,
                      h: float = 0.0001) -> List[float]:
    return [partial_difference_quotient(f, v, i, h)
            for i in range(len(v))]

def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    """
    Moves step_size in the gradients direction, starting on v
    :return: a vector that represents v + step
    """
    step = scalar_multiply(step_size, gradient)
    return add(v, step)

def sum_of_squares_gradient(v: Vector) -> Vector:
    return [2 * v_i for v_i in v]


if __name__ == '__main__':
    rand = random.Random()
    v = [rand.uniform(-10, 10) for _ in range(3)]

    for epoch in range(1000):
        grad = sum_of_squares_gradient(v)
        v = gradient_step(v, grad, -0.01)
        print(epoch, v)