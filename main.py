# Input 
    # Input parametes 
t_start = 0
    # Input Tubrie parameters 
    # Input Ct-curve 
    # Input wind time series 

# Calculate tower and blade deflections 
# Calculate the mean and standard deviation of deflections


import os
import glob
import codecamp
import numpy as np

# Get the absolute path of the parent directory
parent_dir = os.path.dirname(__file__)

folderpath_wind_1 = parent_dir + r"\data\wind_TI_0.1"
folderpath_wind_2 = parent_dir + r"\data\wind_TI_0.05"
folderpath_wind_3 = parent_dir + r"\data\wind_TI_0.15"
folderpaths = [folderpath_wind_1, folderpath_wind_2, folderpath_wind_3]

path_param = parent_dir + r"\data\turbie_parameters.txt"
path_ct = parent_dir + r"\data\CT.txt"



wind_data_list = []
    

for i in folderpaths: 
    # Extract turbulence intensity from folder name
    turbulence_intensity  = None
    folder_name = os.path.basename(i)
    print(f"Folder name is: {folder_name}")
    try:
        turbulence_intensity = float(folder_name.split("_TI_")[1])
    except (IndexError, ValueError):
        print(f"Unable to extract turbulence intensity from folder: {folder_name}")

    # Find all txt files matching the pattern
    file_pattern = os.path.join(i, "wind_*_ms_TI_*.txt")
    files = glob.glob(file_pattern)

    for file in files:
        # Extract wind speed from filename
        filename = os.path.basename(file)
        print(f"File name is: {filename}")
        try:
            wind_speed = float(filename.split("_")[1])
            t_wind, u_wind, x_b, x_t = codecamp.simulate_turbie(filename, path_param, path_ct)

            # mean and standard deviation over t 
            x_b, x_t
            # Convert t_wind and u_wind into a list of (time, velocity) tuples
            data = np.array(zip(turbulence_intensity, wind_speed, t_wind, u_wind, x_b, x_t))
        except (IndexError, ValueError):
            print(f"Skipping file {filename}: unable to extract wind speed")
            continue
    
    # Create WindData object
    wind_data_list.append(data)
    


    
