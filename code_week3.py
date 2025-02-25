"""Script for the Week 3 assignment."""
import codecamp
import os

# Data selection ______________________________________________

# Get the absolute path of the parent directory
parent_dir = os.path.dirname(__file__)

filename_resp = parent_dir + r"\data\resp_12_ms_TI_0.1.txt"
filename_wind = parent_dir + r"\data\wind_12_ms_TI_0.1.txt"

# Input ______________________________________________

# in s
t_resp_start = 60 
t_wind_start = 0 

# Calc _________________________________________________

# Call load_resp and load_wind with the correct relative path
codecamp.load_resp(filename_resp,t_resp_start)
codecamp.load_wind(filename_wind,t_wind_start)

