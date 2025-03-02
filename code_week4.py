"""Script for the Week 4 assignment."""
import codecamp
import os

parent_dir = os.path.dirname(__file__)
# W4P2: Calculate the interpolated thrust coefficient (Ct) for the mean wind velocity.
# filenames
filename_ct = parent_dir + r"\data\CT.txt"
filename_wind = parent_dir + r"\data\wind_12_ms_TI_0.1.txt"

# Call load_wind for u_wind
t_wind, u_wind = codecamp.load_wind(filename_wind, t_start=0)

# Compute the CT value
ct = codecamp.calculate_ct(u_wind, filename_ct)
print(f"Computed CT value ct = {ct:.3f}")
