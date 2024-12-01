import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.special import hermite
import scienceplots

plt.style.use('science')
def psi_n(x, n, hbar, m, omega):
    alpha =  m * omega / hbar
    xi = x * np.sqrt(alpha)
    H_n = hermite(n)
    normalization = np.sqrt(1 / (2**n * sp.factorial(n) * np.sqrt(np.pi)))
    return normalization * H_n(xi) * np.exp(-0.5 * xi**2)


def E_n(n):
    return hbar * omega * (n + 0.5)

def psi_lam(lam, x, t, N, hbar, m, omega):
    res_array = np.zeros_like(x, dtype=complex)
    for i in range(len(N)):
        res_array += np.exp(-0.5 * lam**2) * lam**i * np.exp(-E_n(i) * t * 1j / hbar) / np.sqrt(sp.factorial(i)) * psi_n(x, i, hbar, m, omega)
    return res_array

def probability_density(lam, x, t, N, hbar, m, omega):
    return np.abs(psi_lam(lam, x, t, N, hbar, m, omega))**2

def plot_quantum_probabilities(lam, x, T, N, hbar, m, omega, labels, linestyles, colors):
    for i in range(len(T)):
        plt.plot(x, probability_density(lam, x, T[i], N, hbar, m, omega), label=labels[i], linestyle=linestyles[i], color=colors[i])
    plt.grid()
    plt.legend(edgecolor='black', facecolor='white')
    plt.title('Time evolution of the coherent state')
    plt.xlabel('x')
    plt.ylabel(r'$|\langle x|\lambda, t \rangle|^2$')
    plt.savefig('quantum_probabilities.png', dpi=600)
    plt.show()

m = 1
omega = 1
hbar = 1
x = np.linspace(-5, 5, 1000)
N = np.arange(50)
T = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
labels = [r'$t=0$', r'$t=\frac{\pi}{2}$', r'$t=\pi$', r'$t=\frac{3\pi}{2}$']
linestyles = ['-', '-', '-', ':']
colors = ['b', 'orange', 'g', 'black']

plot_quantum_probabilities(1, x, T, N, hbar, m, omega, labels, linestyles, colors)