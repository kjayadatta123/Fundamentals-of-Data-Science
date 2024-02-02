import numpy as np


time_intervals = np.array([0, 1, 2, 3, 4])  
vertical_positions = np.array([0, 5, 20, 45, 80])

delta_positions = np.diff(vertical_positions)
delta_times = np.diff(time_intervals)
average_velocity = np.mean(delta_positions / delta_times)

print("Average Velocity:", average_velocity, "m/s")
