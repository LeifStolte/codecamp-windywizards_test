"""Script for the Week 3 assignment."""
import os
import codecamp


# Data selection ______________________________________________

# Get the absolute path of the parent directory

filename_resp = r"..\data\resp_12_ms_TI_0.1.txt"
filename_wind = r"..\data\wind_12_ms_TI_0.1.txt"
filename_param = r"..\data\turbie_parameters.txt"

# Input ______________________________________________

# in s
t_resp_start = 60 
t_wind_start = 0 

# Calc _________________________________________________

t, u, xb, xt = codecamp.load_resp(filename_resp,t_resp_start)
t_wind, u_wind = codecamp.load_wind(filename_wind,t_wind_start)
codecamp.load_wind(filename_wind,t_wind_start)

# Plot _____________________________________________________

codecamp.plot_resp(t, u, xb, xt)


# Parameter and matrix ____________________________________

parameters = codecamp.load_turbie_parameters(filename_param)
M, C, K = codecamp.get_turbie_system_matrices(filename_param)
