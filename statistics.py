from collections import Counter
from Math.LinearAlg import sum_of_squares, dot
from typing import List
import math

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)

def median_odd(xs: List[float]) -> float:
    return sorted(xs)[len(xs) // 2]

def median_even(xs:List[float]) -> float:
    sorted_xs = sorted(xs)
    midpoint = len(xs) // 2
    return (sorted_xs[midpoint - 1] + sorted_xs[midpoint]) / 2

def median(xs: List[float]) -> float:
    return median_odd(xs) if len(xs) % 2 == 1 else median_even(xs)

def quantile(xs: List[float], p: float) -> float:
    p_index = int(p * len(xs))
    return sorted(xs)[p_index]

def mode(x:List[float]) -> List[float]:
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items()
            if count == max_count]

def de_mean(xs: List[float]) -> List[float]:
    x_bar = mean(xs)
    return [x - x_bar for x in xs]

def variance(xs: List[float]) -> float:
    n = len(xs)
    deviations = de_mean(xs)
    return sum_of_squares(deviations) / (n-1)

def standard_deviation(xs: List[float]) -> float:
    return math.sqrt(variance(xs))

def interquantile_range(xs: List[float]) -> float:
    '''
    Basically a standard deviation that isn't as affected by outliers
    '''
    return quantile(xs, 0.75) - quantile(xs, 0.25)

def covariance(xs: List[float], ys: List[float]) -> float:
    assert len(xs) == len(ys), "xs and ys must have the same length"
    return dot(de_mean(xs), de_mean(ys)) / len(xs) - 1

def correlation(xs: List[float], ys: List[float]) -> float:
    '''Sempre vai ficar entre -1(anticorrelação perfeita) e +1(correlação perfeita)'''
    stdev_x = standard_deviation(xs)
    stdev_y = standard_deviation(ys)

    if stdev_x > 0 and stdev_y > 0:
        return covariance(xs, ys) / stdev_x / stdev_y
    else:
        return 0

