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


import fastAiry as fa

mp.dps = 30
precision = fa.c(1e-10)

def ai(x, n):
    return fa.airy_mac(x, n)[0]
def bi(x, n):
    return fa.airy_mac(x, n)[1]


x_range = np.linspace(-4, 3, 200)
A_i_ref = mp.matrix(1, x_range.shape[0])
B_i_ref = mp.matrix(1, x_range.shape[0])
A_i = mp.matrix(1, x_range.shape[0])
B_i = mp.matrix(1, x_range.shape[0])
A_i_n = mp.matrix(1, x_range.shape[0])
B_i_n = mp.matrix(1, x_range.shape[0])


for i,x in enumerate(x_range):
    x_mp = fa.c(x)
    A_i_ref[i] = mp.airyai(x_mp)
    B_i_ref[i] = mp.airybi(x_mp)
    A_i[i], A_i_n[i] = fa.optimize_relative(x_mp, ai, A_i_ref[i], precision)
    B_i[i], B_i_n[i] = fa.optimize_relative(x_mp, bi, B_i_ref[i], precision)

plt.plot(x_range, A_i, label='Ai')
plt.plot(x_range, B_i, label='Bi')
plt.plot(x_range, A_i_ref, label='Ai ref', linestyle='dotted', color='red',)
plt.plot(x_range, B_i_ref, label='Bi ref', linestyle='dotted', color='black')
plt.title('Maclaurinov približek in referenčna vrednost')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()

# Create a figure and two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

# Plot the first graph on the top subplot
ax1.set_title('Relativna napaka - Maclaurin')
ax1.plot(x_range, np.abs((A_i - A_i_ref)), label='Ai')
ax1.plot(x_range, np.abs((B_i - B_i_ref)), label='Bi', linestyle='--')
ax1.set_ylabel('Relat. napaka')
ax1.grid()
ax1.legend()

# Plot the second graph on the bottom subplot
ax2.set_title('Število členov za fiksno relat. napako - Maclaurin')
ax2.plot(x_range, A_i_n, label='Ai')
ax2.plot(x_range, B_i_n, label='Bi', linestyle='--')
ax2.set_xlabel('x')
ax2.set_ylabel('Število členov n')
ax2.grid()
ax2.legend()

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()
