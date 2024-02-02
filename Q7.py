import numpy as np

house_data = np.array([
    [3, 1500, 250000],
    [4, 1800, 300000],
    [5, 2000, 350000],
    [3, 1600, 270000],
    [5, 2200, 400000],
    [4, 1900, 320000],
])

houses_more_than_four_bedrooms = house_data[house_data[:, 0] > 4]

average_sale_price_more_than_four_bedrooms = np.mean(houses_more_than_four_bedrooms[:, -1])

print("Average Sale Price of Houses with More than Four Bedrooms:", average_sale_price_more_than_four_bedrooms)
