import numpy as np

fuel_efficiency = np.array([25, 30, 28, 22, 35])

average_efficiency = np.mean(fuel_efficiency)

model1_index = 0
model2_index = 1

percentage_improvement = ((fuel_efficiency[model2_index] - fuel_efficiency[model1_index]) / fuel_efficiency[model1_index]) * 100

print("Average Fuel Efficiency:", average_efficiency)
print(f"Percentage Improvement between Model {model1_index} and Model {model2_index}:", percentage_improvement)
