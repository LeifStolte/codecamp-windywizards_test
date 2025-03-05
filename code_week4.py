"""Script for the Week 4 assignment."""
import codecamp
import os
import numpy as np
import os

# Get the absolute path of the parent directory
parent_dir = os.path.dirname(__file__)

filename_resp = parent_dir + r"\data\resp_12_ms_TI_0.1.txt"
filename_wind = parent_dir + r"\data\wind_12_ms_TI_0.1.txt"
filename_param = parent_dir + r"\data\turbie_parameters.txt"
filename_ct = parent_dir + r"\data\CT.txt"
filename_save = parent_dir + r"\resp\test_resp.txt"

# Input ______________________________________________

# in s
t_resp_start = 60 
t_wind_start = 0 
t_test = 1 # time of the wind speed
y = [1, 2, 3, 4]

# Load Data ____________________________________________

t, u, xb, xt = codecamp.load_resp(filename_resp,t_resp_start)
t_wind, u_wind = codecamp.load_wind(filename_wind,t_wind_start)
parameters = codecamp.load_turbie_parameters(filename_param)
rotor_Dr = parameters.get("Dr")
rho = parameters.get("rho")

# Calc _________________________________________________

# Compute the CT value
ct = codecamp.calculate_ct(u_wind, filename_ct)
rotor_area = np.pi * (rotor_Dr / 2) ** 2
M, C, K = codecamp.get_turbie_system_matrices(filename_param)
dydt = codecamp.calculate_dydt(t_test, y, M, C, K, rho, ct, rotor_area, t_wind, u_wind)
t2, u2, xb, xt = codecamp.simulate_turbie(filename_wind, filename_param, filename_ct)
codecamp.save_resp(t=t2,u=u2,xb=xb, xt=xt, path_save=filename_save) #save at the path 
# Plot _____________________________________________________

#codecamp.plot_resp(t, u, xb, xt)
codecamp.plot_resp(t2, u2, xb, xt)

# Results __________________________________________________

print(f"Computed CT value ct = {ct:.3f}")
print("Calculated dydt =", dydt)
print(f"Simulated turbie response: t = {t2}, u = {u2}, xb = {xb}, xt = {xt}")
print("Done")

