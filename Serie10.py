# Quantum and classical densities for the harmonic oscillator plotting .
# Use as template (to -do: complete missing parts ).
# Step 1: Import Libraries

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite, gammaln

def log_factorial(n):
    return gammaln(n + 1)

# Step 2: Define Key Functions
# Define a function to compute the amplitude (A) based on quantum energy levels.

def compute_amplitude(n, hbar, m, omega):

# Compute amplitude A using E_n = hbar * omega * (n + 1/2)
# and E = 1/2 * m * omega ^2 * A^2.

    En = hbar * omega * (n + 0.5)
    A = np.sqrt(2 * En / (m * omega**2))

    return A # Complete this formula

# Define the quantum wavefunction using Hermite polynomials.
def psi_n(x, n, hbar, m, omega):
# Compute the quantum wavefunction for the nth state .

    alpha =  m * omega / hbar# Scaling factor (involves m, omega , hbar)
    xi = np.linspace(-5, 5, 1000) * np.sqrt(alpha) # Dimensionless variable
    H_n = hermite(n) # Hermite polynomial
    log_normalization = (1/4) * np.log(alpha / np.pi) - (1/2) * (n * np.log(2) + log_factorial(n))
    normalization = np.exp(log_normalization)
    return normalization * H_n(xi) * np.exp(-0.5 * xi**2)

#Define the classical probability density function.
def classical_density(x, A, omega):
    density = np.zeros_like(x)
    for i, xi in enumerate(x):
        if abs(xi) < A:
            density[i] = 1 / (omega * np.sqrt(A**2 - xi**2))
    return density

# Step 3: Parameters and Plotting
hbar = 1.0
m = 1.0
omega = 1.0
n_values = [20, 50, 100] # Quantum states to explore

def plot_densities(n_values, hbar, m, omega):

    # Plot quantum and classical densities for a given quantum state (n).

    # Step 1: Compute amplitude

    rows = len(n_values) // 2 + len(n_values) % 2
    fig, axes = plt.subplots(rows, 2, figsize=(12, 6 * rows))
    axes = axes.flatten()

    for idx, n in enumerate(n_values):
        ax = axes[idx]
    
        A = compute_amplitude(n, hbar, m, omega)
    
    # Step 2: Define x ranges

        x_quantum = np.linspace(-1.5*A, 1.5*A, 1000) # Wider range for quantum
        x_classical = np.linspace(-A, A, 1000) # Limited range for classical
    
    # Step 3: Compute densities

        psi = psi_n(x_quantum, n, hbar, m, omega) # Call psi_n (x, n, hbar , m, omega )
        quantum_density = abs(psi)**2 # Normalize the quantum density
        classical_density_values = classical_density(x_classical, A, omega)
        classical_density_values /= np.trapz(classical_density_values, x_classical) # Normalize the classical density

    # Step 4: Plot

        ax.plot(x_quantum, quantum_density, linestyle="-", linewidth =2, label =f"Quantum $|\\psi_{{n={n}}}(x)|^2$", color ="blue")
        ax.plot(x_classical, classical_density_values, linewidth =2, label =f"Classical $P_{{\\mathrm{{classical}}}}(x)$ (A = {A:.2f})", color ="red")
        ax.axvline(-A, color ="black", linestyle ="--", label =f"Classical Turning Points $x = \\pm{A:.2f}$")
        ax.axvline(A, color ="black", linestyle ="--")

    # Titles and Labels

        ax.set_title(f"Quantum and Classical Densities for $n={n}$ and $\\hbar = {hbar}$")
        ax.set_xlabel("$x$ (Position)")
        ax.set_ylabel("Probability Density")
        ax.legend()
        ax.grid()

    for idx in range(len(n_values), len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle("Quantum and Classical Densities for Various $n$", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Aggiusta layout con spazio per il titolo
    fig.subplots_adjust(hspace=0.4)  # Aumenta lo spazio verticale tra i subplot
    plt.savefig('quantum_classical_densities.png')
    plt.show()

    # Step 4: Plot for Multiple Values of n


n_values = np.array([0, 1, 2, 5, 10, 20, 50, 100])
plot_densities(n_values, hbar, m, omega)