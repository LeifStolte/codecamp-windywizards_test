"""Turbie functions.
"""
import numpy as np

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

def load_turbie_parameters(path):
    processed_data = []
    parameter_names = []

    with open(path,'r') as file:
        lines = file.readlines()

    for line in lines:
        if not line[0] == "%":
            parts = line.strip().split('%')
            parts2 = parts[1].strip().split()
            if len(parts) == 2:
                processed_data.append(float(parts[0].strip()))
                parameter_names.append(parts2[0])

    parameters = dict(zip(parameter_names,processed_data))
    return parameters


def get_turbie_system_matrices(path):
    """Return M, C, K from a txt file path that contains the parameters names and values"""
    parameters = load_turbie_parameters(path)
    m1 = parameters['mb'] 
    m2 = parameters['mn'] + parameters['mt'] + parameters['mh']
    c1 = parameters['c1']
    c2 = parameters['c2']
    k1 = parameters['k1']
    k2 = parameters['k2']
    M = np.array([[m1, 0],[0, m2]])
    C = np.array([[c1, -c1],[-c1, c1+c2]])
    K = np.array([k1, -k2], [-k1, k1+k2])
    return M, C, K

