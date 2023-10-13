import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import mpmath as mp
# Maclaurent series for the Airy functions
# this is a function that finds the n that give approximation of airy_mac that is within the tolerance

def find_n(x, tol):
    n = 1
    while True:
        A_i, B_i = airy_mac(x, n)
        A_i_ref, _, B_i_ref, _ = sp.special.airy(x)
        err = np.abs(A_i - A_i_ref)
        if err < tol:
            break
        n += 1
    return n



def f_mac(x, n):
    k = np.arange(1, n+1, 1)
    k_series = 1 * np.cumprod(x**3 / (3*k * (3*k - 1)))
    return np.sum(k_series) + 1

def g_mac(x, n):
    k = np.arange(1, n+1, 1)
    k_series = x * np.cumprod(x**3 / (3*k * (3*k + 1)))
    return np.sum(k_series) + x


def airy_mac(x_range, n=100):
    alpha = 0.355028053887817239
    beta = 0.258819403792806798
    A_i = np.zeros(len(x_range), dtype=float)
    B_i = np.zeros(len(x_range), dtype=float)
    for i, x in enumerate(x_range):
        f = f_mac(x, n)
        g = g_mac(x, n)
        np.put(A_i, i, alpha * f - beta * g)
        np.put(B_i, i, np.sqrt(3) * (alpha * f + beta * g))
    return [A_i, B_i]
""""

lim_x = 1
lim_y = -6
x_range = np.linspace(-6, 1, 100)
A_i, B_i = airy_mac(x_range)
A_i_ref, _, B_i_ref, _ = sp.special.airy(x_range)
plt.plot(x_range, A_i, label='A_i')
plt.plot(x_range, B_i, label='B_i')
plt.plot(x_range, A_i_ref, label='A_i_ref')
plt.plot(x_range, B_i_ref, label='B_i_ref')
plt.grid()
#plt.ylim(-lim, lim)
plt.legend()
plt.show()
err = np.abs(A_i - A_i_ref)
plt.plot(x_range, err)
plt.show()

"""

def L(z, n):
    k = np.arange(1, n+1, 1)
    # Domnevamo, da bo numpy to naredil hitreje kot Python, če bi uporabili while novi_clen < start_clen:
    k_series = 1 * np.cumprod((3*k - (5/2)) * (3*k - (1/2)) / (18 * z * k))
    return np.sum(k_series) + 1

def P(z, n):
    k = np.arange(1, n+1, 1)
    # Domnevamo, da bo numpy to naredil hitreje kot Python, če bi uporabili while novi_clen < start_clen:
    k_series = 1 * np.cumprod((-1 / (18*z)**2) * (6*k - (11/12)) * (6*k - (7/12)) * (6*k - (5/7)) * (6*k - (1/12)) / (2*k - 1) / 2*k)
    return np.sum(k_series) + 1

def Q(z, n):
    k = np.arange(1, n+2, 1)
    # Domnevamo, da bo numpy to naredil hitreje kot Python, če bi uporabili while novi_clen < start_clen:
    k_series = 1 * np.cumprod((3*k - (5/2)) * (3*k - (1/2)) / (18 * z * k))
    return np.sum(k_series) + 1


def airy_asimp_pos(x_range, n=100):
    ksi_range = (2/3) * np.abs(x_range)**(3/2)
    A_i = np.zeros(len(ksi_range))
    B_i = np.zeros(len(ksi_range))
    for i, ksi in enumerate(ksi_range):
        L_plus = L(ksi, n)
        L_minus = L(-ksi, n)
        np.put(A_i, i, np.e**(-ksi) * L_minus / (2 * np.sqrt(np.pi) * x_range[i]**(1/4)))
        np.put(B_i, i, np.e**(ksi) * L_plus / (np.sqrt(np.pi) * x_range[i]**(1/4)))
    return [A_i, B_i]

x_range = np.linspace(10, 50, 500)
print(x_range)
A_i, B_i = airy_asimp_pos(x_range)
A_i_ref, _, B_i_ref, _ = sp.special.airy(x_range)
print(B_i[-1])
print(B_i_ref[-1])
#plt.plot(x_range, A_i, label='A_i')
plt.plot(x_range, B_i, label='B_i')
#plt.plot(x_range, A_i_ref, label='A_i_ref')
plt.plot(x_range, B_i_ref, label='B_i_ref')
plt.grid()
#plt.ylim(-lim, lim)
plt.legend()
plt.show()
err = np.abs(B_i - B_i_ref)
plt.plot(x_range, err)
plt.show()
