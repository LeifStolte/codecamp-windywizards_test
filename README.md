[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/NbRStOuB)
# CodeCamp WindyWizards Project

## Our CodeCamp project

**Team**: WIndywizards   

(Before submission) Add a brief description here. What does a
user need to know about this code?  

## Quick-start guide
### Prerequisites

Ensure you have the following installed on your system:
- Python 3.10 or later
- Required Python package installations: `numpy`, `matplotlib`, `scipy`

You can install the required packages using pip in terminal:
```sh
pip install numpy matplotlib scipy
```

### Running the Code

1. Ensure you have the necessary data files in the `data` folder:
   - `turbie_parameters.txt`
   - `CT.txt`
   - Wind data files in subfolders `wind_TI_0.1`, `wind_TI_0.05`, and `wind_TI_0.15`

2. Run the main script `main.py`

This will process the wind data, simulate the turbine responses, calculate the mean and standard deviations of the deflections, and plot the results for `wind_TI_0.1`, `wind_TI_0.05`, and `wind_TI_0.15`.

If you want to run it for different turbulence intensities, you can change the folder name in the main.py file fur the functions `run_wind_folder` and `plot_mean_std_comparison`.

## How the code works

### Overview

This project simulates the response in deflection of a wind turbine to different wind conditions and plots the results. The main steps include loading wind data, simulating the turbine response, calculating mean and standard deviation, and plotting the results.

### Code Structure

- `main.py`: The main script that orchestrates the entire process.
- `codecamp/__init__.py`: Contains all the functions used in the project.

### Key Functions

1. **Loading Data**
   - `load_resp(path_resp, t_start)`: Loads response data from a file.
   - `load_wind(path_resp, t_start)`: Loads wind data from a file.
   - `load_turbie_parameters(path)`: Loads turbine parameters from a file.
   - `get_turbie_system_matrices(path)`: Returns the system matrices (M, C, K) for the turbine.

2. **Simulation**
   - `simulate_turbie(path_wind, path_parameters, path_Ct, t_wind_start)`: Simulates the turbine response to wind data.
   - `calculate_dydt(t, y, M, C, K, rho, ct, rotor_area, t_wind, u_wind)`: Computes the time derivative of the state vector.

3. **Statistical Analysis**
   - `mean_standart_deviation(t, u_wind, xb, xt)`: Calculates the mean and standard deviation of the deflections.

4. **Plotting**
   - `plot_resp(t, u, xb, xt)`: Plots the velocity and deflections.
   - `plot_mean_std_comparison(data1, data2, data3)`: Plots the comparison of mean and standard deviations for different turbulence intensities.

5. **Utility Functions**
   - `retrieve_wind_speed_TI(filename)`: Retrieves wind speed and turbulence intensity from the filename.
   - `save_resp(t, u, xb, xt, path_save)`: Saves the response data to a text file.
   - `run_wind_folder(folderpath_wind, path_param, path_ct, t_wind_start)`: Processes all wind data files in a folder and returns the statistical metrics.

### Key Variables

- `t_wind_start`: Start time for the evaluation of the wind data

### Workflow

1. **Data Loading**: The wind data and turbine parameters are loaded from the respective files.
2. **Simulation**: The turbine response is simulated using the loaded data.
3. **Statistical Analysis**: The mean and standard deviations of the blade and tower deflections are calculated.
4. **Plotting**: The results are plotted for visualization.

### Benchmark execution time

Execution time for the entire process is approximately 6 minutes on a standard laptop for all turbulence intensities. Each individual wind data file takes around 5 seconds to process.


## Team contributions

**Best Team Wizards**: Leif Stolte, Lara Schmidt and François de Weck
**Who did what?**:  Workload was distributed equally with a lot of coprogramming on one screen.
For example in week 4, we worked on the following tasks:
- Part 1: Individual
- Part 2: Lara
- Part 3: Leif
- Part 4: François
- Part 5: together
```


