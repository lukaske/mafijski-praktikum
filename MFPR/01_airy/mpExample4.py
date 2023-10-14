import numpy as np
from mpmath import mp
import matplotlib.pyplot as plt
import fastAiry as fa

mp.dps = 120
precision = fa.c(1e-10)

def ai(x, n):
    A_i, _ = fa.airy_asimp_pos(x, n, False)
    return A_i
def bi(x, n):
    _, B_i = fa.airy_asimp_pos(x, n, False)
    return B_i


x_range = np.linspace(8, 20, 200)

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
    A_i[i], A_i_n[i] = fa.optimize(x_mp, ai, A_i_ref[i], precision, 250)
    B_i[i], B_i_n[i] = fa.optimize(x_mp, bi, B_i_ref[i], precision, 250)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

# Plot A_i and its reference on the first subplot (ax1)
ax1.plot(x_range, A_i, label='Ai')
ax1.plot(x_range, A_i_ref, label='Ai ref', linestyle='dotted', color='red')
ax1.set_title('Asimptotski približek in referenca za Ai')
ax1.set_ylabel('y')
ax1.legend()
ax1.grid()

# Plot B_i and its reference on the second subplot (ax2)
ax2.plot(x_range, B_i, label='Bi', color='tab:orange')
ax2.plot(x_range, B_i_ref, label='Bi ref', linestyle='dotted', color='black')
ax2.set_title('Asimptotski približek in referenca za Bi')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.legend()
ax2.grid()

# Adjust spacing between subplots
plt.tight_layout()

# Show the figure with both subplots
plt.show()


# Create a figure and two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

# Plot the first graph on the top subplot
ax1.set_title('Abs. napaka - Asimptotski, x > 0')
ax1.plot(x_range, np.abs(A_i_ref - A_i), label='Ai')
ax1.plot(x_range, np.abs(B_i_ref - B_i), label='Bi', linestyle='--')
ax1.set_ylabel('Abs. napaka')
ax1.grid()
ax1.legend()

# Plot the second graph on the bottom subplot
ax2.set_title('Število členov za fiksno abs. napako - Maclaurin')
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
