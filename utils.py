import math


def lcm(a, b):
    return abs(a*b) // math.gcd(a, b)

