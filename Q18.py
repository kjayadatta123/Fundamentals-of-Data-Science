import numpy as np

temperature_data = [
    [20, 22, 25, 19, 18, 23, 21, 24, 20, 22, 25, 19],
    [22, 24, 26, 20, 19, 24, 22, 25, 21, 23, 26, 20],
]

means = [np.mean(temperatures) for temperatures in temperature_data]
for i, mean in enumerate(means):
    print(f"City {i + 1}: Mean Temperature = {mean:.2f}°C")

std_devs = [np.std(temperatures) for temperatures in temperature_data]
for i, std_dev in enumerate(std_devs):
    print(f"City {i + 1}: Standard Deviation = {std_dev:.2f}°C")

ranges = [np.max(temperatures) - np.min(temperatures) for temperatures in temperature_data]
city_highest_range = np.argmax(ranges)
print(f"City {city_highest_range + 1} has the highest temperature range.")

most_consistent_city = np.argmin(std_devs)
print(f"City {most_consistent_city + 1} has the most consistent temperature.")
