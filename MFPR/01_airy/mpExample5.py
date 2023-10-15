import mpmath
from mpmath import mp
import numpy as np
import matplotlib.pyplot as plt

mp.dps = 60
mp.pretty = True

x_range = np.arange(5, 51, 5)

def fr(x,y):
    return mp.mpf(x) / mp.mpf(y)
def mu(x,y):
    return mp.mpf(x) * mp.mpf(y)

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


x_range = np.linspace(-20, -15, 200)
A_i = mp.matrix(1, x_range.size)
B_i = mp.matrix(1, x_range.size)
A_i_ref = mp.matrix(1, x_range.size)
B_i_ref = mp.matrix(1, x_range.size)


for i, x in enumerate(x_range):
    A_i[i], B_i[i] = airy_asimp_neg(x)
    A_i_ref[i], B_i_ref[i] = mp.airyai(x), mp.airybi(x)

plt.plot(x_range, A_i, label="A_i")
plt.plot(x_range, A_i_ref, label="A_i ref", linestyle='dotted')

plt.plot(x_range, B_i, label="B_i")
plt.plot(x_range, B_i_ref, label="B_i ref", linestyle='dotted')

plt.legend()
plt.show()

plt.plot(x_range, np.abs(A_i - A_i_ref), label="error")
plt.legend()
plt.show()

1/0
x = -5
natancnost = np.arange(1, 22, 1)
napake = mp.matrix(1, natancnost.size)
for i, n in enumerate(natancnost):
    A_i, B_i = airy_asimp_neg(x, n)
    A_i_ref, B_i_ref = mp.airyai(x), mp.airybi(x)
    print(n, B_i_ref - B_i)
    napake[i] = np.abs(B_i_ref - B_i)

plt.plot(natancnost, napake)
plt.yscale('log')
plt.show()


