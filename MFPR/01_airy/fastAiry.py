import numpy as np
import scipy as scp
from mpmath import mp

def c(x):
    return mp.mpf(x)

def f_mac_np(x, n):
    k = np.arange(1, n+1, 1)
    k_series = 1 * mp.cumprod(x**3 / (3*k * (3*k - 1)))
    return np.sum(k_series) + 1

def g_mac_np(x, n):
    k = np.arange(1, n+1, 1)
    k_series = x * np.cumprod(x**3 / (3*k * (3*k + 1)))
    return np.sum(k_series) + x

def f_mac(x, n):
    sum = mp.mpf(1)
    term = mp.mpf(1)
    i = 1
    while (i <= n):
        k = mp.mpf(int(i))
        term *= (x**3 / (3*k * (3*k - 1)))
        sum += term
        i += 1
    return sum

def g_mac(x, n):
    sum = mp.mpf(x)
    term = mp.mpf(x)
    i = 1
    while (i <= n):
        k = mp.mpf(int(i))
        term *= (x**3 / (c(3)*k * (c(3)*k + c(1))))
        sum += term
        i += 1
    return sum

def L(z, max_iter=100, brk=True):
    sum = mp.mpf(1)
    term = mp.mpf(1)
    i = 1
    while (i < max_iter):
        k = mp.mpf(int(i))
        new_term = term * (3 * k - (mp.mpf(5) / mp.mpf(2))) * (3 * k - (mp.mpf(1) / mp.mpf(2))) / (18 * z * k)
        if (np.abs(term) < np.abs(new_term)):
            if brk: break
        term = new_term
        sum += term
        i += 1
    return sum

def airy_asimp_pos(x, n=150, brk=True):
    x = mp.mpf(x)
    ksi = mp.mpf(2) / mp.mpf(3) * mp.sqrt(x**3)
    L_plus = L(ksi, n, brk)
    L_minus = L(-ksi, n, brk)
    A_i = mp.e**(-ksi) * L_minus / (2 * np.sqrt(np.pi) * x**(1/4))
    B_i = mp.e**(ksi) * L_plus / (mp.sqrt(mp.pi) * mp.sqrt(mp.sqrt(x)))
    return [A_i, B_i]

def airy_mac(x, n=100):
    n = c(n)
    x = c(x)
    alpha = c(0.355028053887817239)
    beta = c(0.258819403792806798)
    f = f_mac(x, n)
    g = g_mac(x, n)
    return [alpha * f - beta * g, mp.sqrt(3) * (alpha * f + beta * g)]

def optimize(x, f, ref_value, target_err, max_iter=100):
    n = 1
    g = f(x, n)
    while n <= max_iter:
        g = f(x, n)
        err = np.abs(g - ref_value)
        if err < target_err:
            break
        n += 1
    return [g, n]

def optimize_relative(x, f, ref_value, target_err, max_iter=100):
    n = 1
    g = f(x, n)
    while n <= max_iter:
        g = f(x, n)
        err = np.abs(np.abs(g - ref_value) / ref_value) 
        if err < target_err:
            break
        n += 1
    return [g, n]