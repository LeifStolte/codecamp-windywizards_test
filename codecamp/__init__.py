"""Turbie functions.
"""
import numpy as np
import matplotlib.pyplot as plt

def load_resp(path_resp, t_start = 60):
    """An example function in a package."""
    # Load the data, skipping the header row
    data = np.loadtxt(path_resp, skiprows=1)
    #create a start and end value for loading our data
    i = 0
    while data[i,0] < t_start:
        i += 1
    # Extract columns into separate NumPy arrays
    t = data[i:, 0]  # First column: Time(s)
    u = data[i:, 1]  # Second column: V(m/s)
    xb = data[i:, 2]  # Third column: xb(m)
    xt = data[i:, 3]  # Fourth column: xt(m)

    # Print first few values for verification
    print("t:", t[:5])
    print("u:", u[:5])
    print("xb:", xb[:5])
    print("xt:", xt[:5])

    return t, u, xb, xt

def load_wind(path_resp, t_start = 0):
    """An example function in a package."""
    # Load the data, skipping the header row
    data = np.loadtxt(path_resp, skiprows=1)
    #create a start and end value for loading our data
    i = 0
    while data[i,0] < t_start:
        i += 1
    # Extract columns into separate NumPy arrays
    t_wind = data[i:, 0]  # First column: Time(s)
    u_wind = data[i:, 1]  # Second column: V(m/s)

    return t_wind, u_wind

def plot_resp(t, u, xb, xt):
    
    # Create subplots with improved spacing
    fig, axs = plt.subplots(3, 1, figsize=(9, 4))

    # First plot
    axs[0].plot(t, u, color='tab:blue', linewidth=0.5)
    axs[0].set_title('U over Time', fontsize=12, fontweight='bold')
    axs[0].set_xlabel('$t$ in [s]', fontsize=10)
    axs[0].set_ylabel('$u(t)$ in [m/s]', fontsize=10)
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].set_xlim(60,600)

    # Second plot
    axs[1].plot(t, xb, color='tab:blue', linewidth=0.5)
    axs[1].set_title('Xb over Time', fontsize=12, fontweight='bold')
    axs[1].set_xlabel('$t$ in [s]', fontsize=10)
    axs[1].set_ylabel('$xb(t)$ in [m]', fontsize=10)
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].set_xlim(60,600)

    # Third plot
    axs[2].plot(t, xt, color='tab:blue', linewidth=0.5)
    axs[2].set_title('Xt over Time', fontsize=12, fontweight='bold')
    axs[2].set_xlabel('$t$ in [s]', fontsize=10)
    axs[2].set_ylabel('$xt(t)$ in [m]', fontsize=10)
    axs[2].grid(True, linestyle='--', alpha=0.6)
    axs[2].set_xlim(60,600)

    # Show the plot
    fig.tight_layout()
    plt.show()
    