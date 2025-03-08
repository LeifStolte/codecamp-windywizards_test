"""Goes through the data folder and extracts the wind data from the files in the folder
    Calculates the Ct and Ct curve for each wind speed
    Calculates the tower and blade deflections for each wind speed 
    Calculates the mean and standard deviation of the deflections
    Plots the wind data in a single plot"""





import os
import glob
import codecamp
import numpy as np
import matplotlib as plt

# Set the start time of the simulation

t_wind_start = 0 

#retrieve the wind data from the data folder#
# Get the absolute path of the parent directory
parent_dir = os.path.dirname(__file__)

folderpath_wind_1 = parent_dir + r"\data\wind_TI_0.1"
folderpath_wind_2 = parent_dir + r"\data\wind_TI_0.05"
folderpath_wind_3 = parent_dir + r"\data\wind_TI_0.15"
folderpaths = [folderpath_wind_1, folderpath_wind_2, folderpath_wind_3]

path_param = parent_dir + r"\data\turbie_parameters.txt"
path_ct = parent_dir + r"\data\CT.txt"

"""
dataTI = []
#loop trough the different tubulance intensities folder
for folderpath in folderpaths:
    wind_data_list = []
    for file in folderpath:
        # Extract turbulence intensity and wind speed from the filename
        wind_speed, turbulence_intensity = codecamp.retrieve_wind_speed_TI(filename=file)
        if wind_speed is None or turbulence_intensity is None:
            continue

        # Simulate the Turbie response
        t, u_wind, x_b, x_t = codecamp.simulate_turbie(path_wind=file, path_parameters=path_param, path_Ct=path_ct, t_start=t_wind_start)

        # Calculate the mean and standard deviation of the deflections
        mean_xb, mean_xt, std_xb, std_xt = codecamp.mean_standart_deviation(t=t, u_wind=u_wind, xb=x_b, xt=x_t)

        #append the data to the list
        wind_data_list.append([wind_speed, turbulence_intensity, mean_xb, mean_xt, std_xb, std_xt])
    #append the data to the list
    #dataTI.append(wind_data_list)

    #plot the wind data creating a new plot for each wind speed
    
"""

dataTI01 = codecamp.run_wind_folder(folderpath_wind_1, path_param, path_ct, t_wind_start)
dataTI005 = codecamp.run_wind_folder(folderpath_wind_2, path_param, path_ct, t_wind_start)
dataTI015 = codecamp.run_wind_folder(folderpath_wind_3, path_param, path_ct, t_wind_start)

print(dataTI01)
print(os.listdir(folderpath_wind_1))
    
codecamp.plot_mean_std_comparison(dataTI01, dataTI005, dataTI015)

