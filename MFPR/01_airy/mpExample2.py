import scipy.integrate as spi
import numpy as np
from mpmath import *

mp.dps = 130
mp.pretty = True

# Define the integrand as a Python function
def integrand(t):
    return mp.exp(-1 * (t**3 / 3) + 20 * t) + mp.sin((t**3 / 3) + 20 * t)


# Perform the numerical integration with infinite bounds
result = quad(integrand, [0, mp.inf]) / mp.pi

print(result)
