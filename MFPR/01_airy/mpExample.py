from mpmath import mp
import numpy as np
import matplotlib.pyplot as plt

mp.dps = 130
mp.pretty = True

x_range = np.arange(5, 51, 5)

def L(z, n):
    sum = mp.mpf(1)
    term = mp.mpf(1)
    i = 1
    while (i < n):
        k = mp.mpf(int(i))
        new_term = term * (3 * k - (mp.mpf(5) / mp.mpf(2))) * (3 * k - (mp.mpf(1) / mp.mpf(2))) / (18 * z * k)
        if (np.abs(term) < np.abs(new_term)):
            pass
        term = new_term
        sum += term
        i += 1
    return sum

def airy_asimp_pos(x, n=500):
    x = mp.mpf(x)
    ksi = mp.mpf(2) / mp.mpf(3) * mp.sqrt(x**3)
    L_plus = L(ksi, n)
    L_minus = L(-ksi, n)
    A_i = mp.e**(-ksi) * L_minus / (2 * np.sqrt(np.pi) * x**(1/4))
    B_i = mp.e**(ksi) * L_plus / (mp.sqrt(mp.pi) * mp.sqrt(mp.sqrt(x)))
    return [A_i, B_i]

x = 20
wolfram = '2.1037650496511038144947890143998843924295089896416'
wolfram_ref = mp.mpf(wolfram) * 10**25
print("Wolfram ref:", wolfram_ref)
natancnost = np.arange(21, 261, 1)
napake = mp.matrix(1, natancnost.size)
for i, n in enumerate(natancnost):
    A_i, B_i = airy_asimp_pos(x, n)
    A_i_ref, B_i_ref = mp.airyai(x), mp.airybi(x)
    print(n, B_i_ref - wolfram_ref)
    napake[i] = np.abs(B_i_ref - wolfram_ref)

plt.plot(natancnost, napake)
plt.yscale('log')
plt.show()