# This script is the main script for the codecamp project. It runs the codecamp module for each wind folder and plots the mean and standard deviation of the deflections for each wind folder.

import os
import src.codecamp as codecamp


# Set the start time for the evaluation of the wind data
T_WIND_START = 0 

# Get the absolute path of the parent directory
parent_dir = os.path.dirname(__file__)

# Get the absolute path of the current file
folderpath_wind_1 = parent_dir + r"\data\wind_TI_0.1"
folderpath_wind_2 = parent_dir + r"\data\wind_TI_0.05"
folderpath_wind_3 = parent_dir + r"\data\wind_TI_0.15"

path_param = parent_dir + r"\data\turbie_parameters.txt"
path_ct = parent_dir + r"\data\CT.txt"

# Run the codecamp module for each wind folder
dataTI01 = codecamp.run_wind_folder(folderpath_wind_1, path_param, path_ct, T_WIND_START)
dataTI005 = codecamp.run_wind_folder(folderpath_wind_2, path_param, path_ct, T_WIND_START)
dataTI015 = codecamp.run_wind_folder(folderpath_wind_3, path_param, path_ct, T_WIND_START)

# Plot the mean and standard deviation of the deflections
codecamp.plot_mean_std_comparison(dataTI01, dataTI005, dataTI015)