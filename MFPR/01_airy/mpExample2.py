import scipy.integrate as spi
import numpy as np
from mpmath import mp
import matplotlib.pyplot as plt
import fastAiry as fa

mp.dps = 30
precision = fa.c(1e-10)

def ai(x, n):
    return fa.airy_mac(x, n)[0]
def bi(x, n):
    return fa.airy_mac(x, n)[1]


x_range = np.linspace(-1, 3, 200)
A_i_ref = mp.matrix(1, x_range.shape[0])
B_i_ref = mp.matrix(1, x_range.shape[0])
A_i = mp.matrix(1, x_range.shape[0])
B_i = mp.matrix(1, x_range.shape[0])
A_i_n = mp.matrix(1, x_range.shape[0])
B_i_n = mp.matrix(1, x_range.shape[0])


for i,x in enumerate(x_range):
    print(i / x_range.shape[0] * 100, '%')
    x_mp = fa.c(x)
    A_i_ref[i] = mp.airyai(x_mp)
    B_i_ref[i] = mp.airybi(x_mp)

    A_i[i], A_i_n[i] = fa.optimize(x_mp, ai, A_i_ref[i], precision)
    print('a_n', A_i_n[i])
    B_i[i], B_i_n[i] = fa.optimize(x_mp, bi, B_i_ref[i], precision)
    print('b_n', B_i_n[i])


plt.plot(x_range, A_i, label='Ai')
plt.plot(x_range, B_i, label='Bi')
plt.plot(x_range, A_i_ref, label='Ai ref')
plt.plot(x_range, B_i_ref, label='Bi ref')
plt.legend()
plt.grid()
plt.show()

plt.plot(x_range, np.abs(A_i_ref - A_i), label='Ai napaka, abs(def. - mpmath)', linestyle='--')
plt.plot(x_range, np.abs(B_i_ref - B_i), label='Bi napaka, abs(def. - mpmath)', linestyle='dotted')
plt.show()


