import mpmath
import numpy as np
from mpmath import mp

def c(x):
    return mp.mpf(x)
def fr(x,y):
    return mp.mpf(x) / mp.mpf(y)
def mu(x,y):
    return mp.mpf(x) * mp.mpf(y)

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

def P(z, n):
    sum = mp.mpf(1)
    term = mp.mpf(1)
    i = 1
    while (i < n):
        k = mp.mpf(int(i))
        new_term = term * ((mu(6,k) - fr(1,2)) * (mu(6,k) - fr(5,2)) * (mu(6,k) - fr(7,2)) * (mu(6,k) - fr(11,2)) ) / (mp.mpf(-324) * z**mp.mpf(2) * mu(2,k) * (mu(2,k) - mu(1,1)))
        if (mpmath.fabs(term) < mpmath.fabs(new_term)):
            break
        term = new_term
        sum += term
        i += 1
    return sum

def Q(z, n):
    sum = mu(fr(1,z),fr(5,72))
    term = sum
    i = 1
    while (i < n):
        k = mp.mpf(int(i))
        new_term = term * ((mu(6,k) - fr(1,2)) * (mu(6,k) - fr(5,2)) * (mu(6,k) + fr(1,2)) * (mu(6,k) + fr(5,2)) ) / (mp.mpf(-324) * z**mp.mpf(2) * mu(2,k) * (mu(2,k) + mu(1,1)))
        if (mpmath.fabs(term) < mpmath.fabs(new_term)):
            break
        term = new_term
        sum += term
        i += 1
    return sum


def airy_asimp_neg(x, n=15):
    x = mp.mpf(x)
    ksi = fr(2,3) * mp.sqrt(mpmath.fabs(x)**mu(3,1))
    Q_mp = Q(ksi, n)
    P_mp = P(ksi, n)
    T = mp.mpf(1) / (mp.sqrt(mp.pi) * mp.sqrt(mp.sqrt(mu(1,-1)*x)))
    fi = mp.pi / mp.mpf(4)
    A_i = T * (mp.sin(ksi - fi) * Q_mp + mp.cos(ksi - fi) * P_mp)
    B_i = T * (mp.cos(ksi - fi) * Q_mp - mp.sin(ksi - fi) * P_mp)

    return mpmath.re(A_i), mpmath.re(B_i)


def airy_asimp_pos(x, n=150, brk=True):
    x = mp.mpf(x)
    ksi = mp.mpf(2) / mp.mpf(3) * mp.sqrt(x**3)
    L_plus = L(ksi, n, brk)
    L_minus = L(-ksi, n, brk)
    A_i = mp.e**(-ksi) * L_minus / (mp.mpf(2) * np.sqrt(np.pi) * mp.sqrt(mp.sqrt(x)))
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

def f_zero(z):
    z = c(z)
    return z**fr(2,3) * (c(1) + fr(5,48) * z**c(-2) - fr(5,36) * z**c(-4) + fr(77125, 82944) * z**c(-6) - fr(108056875, 6967296) * z**c(-8))

def zero_arg(n, k):
    return fr(3, 8) * mp.pi * (mu(4, n) - c(k))