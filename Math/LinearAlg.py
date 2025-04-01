import math
from typing import List

Vector = List[float]

def add(v: Vector, w: Vector) -> Vector:
    return [v_i + w_i for v_i, w_i in zip(v, w)]

def subtract(v: Vector, w: Vector) -> Vector:
    return [v_i - w_i for v_i, w_i in zip(v, w)]

def vector_sum(vectors: List[Vector]) -> Vector:
    num_elements = len(vectors[0])

    return [sum(vector[i] for vector in vectors)
            for i in range(num_elements)]

def scalar_multiply(c: float, v: Vector) -> Vector:
    return [v_i * c for v_i in v]

def vector_mean(vectors: List[Vector]) -> Vector:
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

def distance(v1: Vector, v2: Vector) -> float:
    # dist = sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return sum([(v1i - v2i)**2 for v1i, v2i in zip(v1, v2)])**0.5

def dot(v: Vector, w: Vector) -> float:
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def sum_of_squares(v: Vector) -> float:
    return dot(v, v)

def magnitude(v: Vector) -> float:
    return math.sqrt(sum_of_squares(v))