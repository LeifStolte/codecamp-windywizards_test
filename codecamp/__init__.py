"""Turbie functions.
"""
import numpy as np

def load_resp(path_resp, t_start = 0):
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
    print("Time:", t[:5])
    print("Velocity:", u[:5])
    print("xb:", xb[:5])
    print("xt:", xt[:5])

    return t, u, xb, xt

load_resp(r"C:\Users\fhuer\OneDrive\Dokumente\Lara\GitHub\codecamp-windywizards\data\resp_12_ms_TI_0.1.txt",60)

