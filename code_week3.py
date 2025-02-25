"""Script for the Week 3 assignment."""
import codecamp
import os

# Get the absolute path of the parent directory
parent_dir = os.path.dirname(__file__)

filename_resp = parent_dir + r"\data\resp_12_ms_TI_0.1.txt"
filename_wind = parent_dir + r"\data\wind_12_ms_TI_0.1.txt"
filename_param = parent_dir + r"data\turbie_parameters.txt"

# Call load_resp with the correct relative path


# Input ______________________________________________

# in s
t_resp_start = 60 
t_wind_start = 0 

# Calc _________________________________________________

codecamp.load_resp(filename_resp,t_resp_start)


codecamp.load_wind(filename_wind,t_wind_start)

parameters = codecamp.load_turbie_parameters(filename_param)

M, C, K = codecamp.get_turbie_system_matrices(filename_param)