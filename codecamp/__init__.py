"""Turbie functions.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
import glob
import os

def load_resp(path_resp, t_start = 60):
    """Load the data t, u, xb & xt, skipping the header row
    Input: filename, t_start"
    Output: t, u, xb, xt """
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
    """Load the data u & t, skipping the header row
    Input: filename, t_start
    Output:t_wind, u_wind"""
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
    """from a txt file path that contains the parameters names and values, creates a dictionary
    Input: filename
    Output: parameters (dictionary { Name(str): Values(float)}"""
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
    m1 = 3 * parameters['mb'] 
    m2 = parameters['mn'] + parameters['mt'] + parameters['mh']
    c1 = parameters['c1']
    c2 = parameters['c2']
    k1 = parameters['k1']
    k2 = parameters['k2']
    M = np.array([[m1, 0],[0, m2]])
    C = np.array([[c1, -c1],[-c1, c1+c2]])
    K = np.array([[k1, -k1], [-k1, k1+k2]])
    return M, C, K

def plot_resp(t, u, xb, xt):
    """Plots 3 graphs from  u(t) , xb(t) and xt(t)
    input: t, u, xb, xt (list of float)
    output: None"""
    
    # Create subplots with improved spacing
    fig, axs = plt.subplots(2, 1, figsize=(9, 4))
    axs = tuple(axs)

    # First plot
    axs[0].plot(t, u, color='tab:blue', linewidth=0.5)
    axs[0].set_title('Velocity $u$', fontweight='bold')
    axs[0].set_xlabel('$t$ in [s]')
    axs[0].set_ylabel('$u(t)$ in [m/s]')
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].set_xlim(60,660)

    # Second plot
    axs[1].plot(t, xb, color='tab:blue', label= "$xb$", linewidth=0.5)
    axs[1].plot(t, xt, color='purple', label= "$xt$", linewidth=0.5)
    axs[1].set_title('Deflections $xb$ and $xt$', fontweight='bold')
    axs[1].set_xlabel('$t$ in [s]')
    axs[1].set_ylabel('$x(t)$ in [m]')
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].set_xlim(60,660)
    axs[1].legend()

    # Show the plot
    fig.tight_layout()
    plt.show()
    return fig, axs
    
def calculate_ct(u_wind, path_ct):
    """Calculates the interpolated thrust coefficient (Ct) for the mean wind velocity.
    Input:  u_wind (array): Array of wind velocity values [m/s].
            path_ct (str): Path to the file containing Ct curve data.
    Output: ct (float): Interpolated Ct value corresponding to the mean wind velocity.
    """

    # Load CT data (skipping the header row)
    data = np.loadtxt(path_ct, skiprows=1)
    
    # Extract columns: wind speed and CT curve
    u_ct = data[:, 0]  # First column: wind speed [m/s]
    ct_curve = data[:, 1]  # Second column: thrust coefficient

    # Compute the mean wind speed
    u_mean = np.mean(np.array(u_wind))
    #print(f"Average velocity u_mean = {u_mean:.3f} [m/s]")

    # Interpolate to determine the CT value
    ct = np.interp(u_mean, u_ct, ct_curve)
    return ct

def calculate_dydt(t, y, M, C, K, rho=None, ct=None, rotor_area=None, t_wind=None, u_wind=None):
    """
    Computes the time derivative of the state vector y for a two-degree-of-freedom system.

    Input:
    - t (float): Time 
    - y (numpy array): State vector of shape (4,)
    - M (numpy array): Mass matrix of shape (2,2)
    - C (numpy array): Damping matrix of shape (2,2)
    - K (numpy array): Stiffness matrix of shape (2,2)
    - rho (float, optional): Air density [kg/m^3].
    - ct (float, optional): Thrust coefficient [-]
    - rotor_area (float, optional): Rotor area [m^2]
    - t_wind (numpy array, optional): Time array for wind speed series [s]
    - u_wind (numpy array, optional): Wind speed array [m/s]

    Output: 
    - dydt (numpy array): Time derivative of the state vector y (1D array of shape (4,))
    """
    Ndof = M.shape[0]  # Number of degrees of freedom
    M_inv = np.linalg.inv(M)  # Inverse of the mass matrix
    I = np.eye(Ndof)  # Identity matrix of size Ndof

    # Define the system matrices
    O = np.zeros((Ndof, Ndof))  # Zero matrix of size Ndof x Ndof
    F = np.zeros(Ndof)  # Zero vector of size Ndof
    A = np.block([[O, I], [-M_inv @ K, -M_inv @ C]])  # System matrix A

    # Define the forcing term
    if t_wind is not None:
        x1 = y[2]  # Blade deflection
        u = np.interp(t, t_wind, u_wind)  # Interpolated wind speed at time t
        f_aero = 0.5 * rho * ct * rotor_area * (u - x1) * np.abs(u - x1)  # Aerodynamic force
        F[0] = f_aero  # Apply aerodynamic force to the first degree of freedom

    # Assemble matrices B and dydt
    B = np.concatenate((np.zeros(Ndof), M_inv @ F))  # Forcing vector B
    dydt = A @ y + B  # Time derivative of the state vector
    return dydt.flatten()  # Return the flattened dydt vector

def simulate_turbie(path_wind, path_parameters, path_Ct, t_wind_start=0):
    """
    Simulate homogenous and forced responses by solving an IVP for the given parameter and wind speeds.

    Parameters:
    - path_wind (str or pathlib.Path): Path to the wind data file.
    - path_parameters (str or pathlib.Path): Path to the turbine parameters file.
    - path_Ct (str or pathlib.Path): Path to the Ct curve data file.
    - t_wind_start (float): Start time of the wind data [s].

    Returns:
    - tuple: (t, u_wind, x_b, x_t)
        - t (numpy array): Time of the simulated response [s].
        - u_wind (numpy array): Streamwise wind speed [m/s].
        - x_b (numpy array): Displacement of the blade [m].
        - x_t (numpy array): Displacement of the tower [m].
    """
    # Load wind data
    t_wind, u_wind = load_wind(path_resp=path_wind, t_start=t_wind_start)
    
    # Load turbine parameters
    parameters = load_turbie_parameters(path=path_parameters)
    
    # Extract parameters
    rho = parameters['rho']
    Dr = parameters['Dr']
    rotor_area = np.pi * (Dr / 2) ** 2
    
    # Get system matrices
    M, C, K = get_turbie_system_matrices(path=path_parameters)
    
    # Calculate thrust coefficient
    ct = calculate_ct(u_wind=u_wind, path_ct=path_Ct)
    
    # Define time span and initial conditions
    t_span = (t_wind[0], t_wind[-1])
    initial_cond = np.array([0, 0, 0, 0])
    
     # Solve the initial value problem
    solution = spi.solve_ivp(
        fun=calculate_dydt, 
        t_span=t_span, 
        y0=initial_cond, 
        args=(M, C, K, rho, ct, rotor_area, t_wind, u_wind),
        t_eval=t_wind,  # Ensure the solution is evaluated at the same time points as the wind data
        method='RK23' # Use the Runge-Kutta method 23 to solve the ODE and ensure the solution is evaluated at the same time points as the wind data
    ) 
    # Extract results
    t = t_wind
    y = solution.y
    x_b = y[0, :] - y[1, :] # Blade deflection
    x_t = y[1, :]  # Tower deflection
    
    return t, u_wind, x_b, x_t

def save_resp(t, u, xb, xt, path_save):
    """Save the response data to a text file.
    Input: t, u, xb, xt, path_save
    Output: None"""
    data = np.array([t, u, xb, xt]).T  
    np.savetxt(fname=path_save, X=data, delimiter='\t', fmt='%.3f', header="t\t u\t xb\t xt")
    return None

def retrieve_wind_speed_TI(filename):
    """Retrieve the wind speed and turbulence intensity from the filename.
    Input: filename (str)
    Output: wind_speed (float), turbulence_intensity (float)"""
    
    try:
        # Extract wind speed
        wind_speed = float(filename.split("_")[1])
        
        # Extract turbulence intensity
        turbulence_intensity = float(filename.split("_TI_")[1].split(".")[0])
        
    except (IndexError, ValueError) as e:
        print(f"Error extracting data from filename {filename}: {e}")
        return None, None
    
    return wind_speed, turbulence_intensity

def mean_standart_deviation(t, u_wind, xb, xt):
    """Calculate the mean and standard deviation of the deflections.
    Input: t, u_wind, xb, xt
    Output: mean_xb, mean_xt, std_xb, std_xt"""
    mean_xb = np.mean(xb)
    mean_xt = np.mean(xt)
    std_xb = np.std(xb)
    std_xt = np.std(xt)
    return mean_xb, mean_xt, std_xb, std_xt

#Run function on folder
def run_wind_folder(folderpath_wind, path_param, path_ct, t_wind_start):
    """retrieve the wind data from the data folder
    read parameter of the wind turbine from a parameter filer
    Look into a folder and loop over each file in the folder
        Extract the wind speed and TI from the file name
        Read the wind data from the file (done in simulate_turbie)
        calculate Ct and Ct curve (done in sumilate_turbie)
        calculate the tower and blade deflections (done in simulate_turbie)
        calculate the mean and standard deviation of the deflections
        plot all the wind data in a single plot
        create a  folder to save the responds
        save the reponse in the folder"""
    #create list to save the data
    wind_data_list = np.array([])
    for file in folderpath_wind:
        # Extract turbulence intensity and wind speed from the filename
        wind_speed, turbulence_intensity = retrieve_wind_speed_TI(filename=file)
        if wind_speed is None or turbulence_intensity is None:
            continue

        # Simulate the Turbie response
        t, u_wind, x_b, x_t = simulate_turbie(path_wind=file, path_parameters=path_param, path_Ct=path_ct, t_wind_start=t_wind_start)

        # Calculate the mean and standard deviation of the deflections
        mean_xb, mean_xt, std_xb, std_xt = mean_standart_deviation(t=t, u_wind=u_wind, xb=x_b, xt=x_t)

        #append the data to the list
        wind_data_list.append([wind_speed, turbulence_intensity, mean_xb, mean_xt, std_xb, std_xt])
    return wind_data_list






    

    