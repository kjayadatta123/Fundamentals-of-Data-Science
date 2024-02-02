import numpy as np

sales_data = np.array([
    [100, 2, 30.5],
    [101, 1, 15.75],
    [102, 3, 50.0],
    [103, 1, 10.99],

])

average_price = np.mean(sales_data[:, -1])

print("Average Price of All Products Sold:", average_price)
