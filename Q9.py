import matplotlib.pyplot as plt
import numpy as np

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
sales_values = [5000, 6000, 7500, 8000, 9000]

# a. Line plot of the monthly sales data
plt.figure(figsize=(8, 5))
plt.plot(months, sales_values, marker='o', linestyle='-', color='b')
plt.title('Monthly Sales Data (Line Plot)')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.grid(True)
plt.show()

# b. Bar plot of the monthly sales data
plt.figure(figsize=(8, 5))
plt.bar(months, sales_values, color='green')
plt.title('Monthly Sales Data (Bar Plot)')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.grid(axis='y')
plt.show()
