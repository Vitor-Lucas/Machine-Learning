from typing import List

Vector = List[float]

# dist = sqrt((x1 - x2)**2 + (y1 - y2)**2)

def distance(v1: Vector, v2: Vector) -> float:
    return sum([(v1i - v2i)**2 for v1i, v2i in zip(v1, v2)])**0.5
