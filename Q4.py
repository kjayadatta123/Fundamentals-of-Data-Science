import pandas as pd

# Creating a DataFrame with predefined data
data = {
    'property_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'location': ['City A', 'City B', 'City A', 'City C', 'City B', 'City C', 'City A', 'City B', 'City C', 'City A'],
    'number_of_bedrooms': [3, 4, 2, 5, 3, 4, 2, 3, 4, 5],
    'area_sqft': [1200, 1500, 900, 2000, 1300, 1800, 950, 1200, 1600, 2100],
    'listing_price': [250000, 350000, 200000, 500000, 300000, 450000, 220000, 280000, 400000, 550000]
}

property_data = pd.DataFrame(data)

# Display the predefined DataFrame
print("Predefined Property Data:")
print(property_data)
print("\n")

# Task 1: Average listing price of properties in each location
average_price_by_location = property_data.groupby('location')['listing_price'].mean()
print("1. Average listing price of properties in each location:\n", average_price_by_location)
print("\n")

# Task 2: Number of properties with more than four bedrooms
num_properties_more_than_four_bedrooms = len(property_data[property_data['number_of_bedrooms'] > 4])
print("2. Number of properties with more than four bedrooms:", num_properties_more_than_four_bedrooms)
print("\n")

# Task 3: Property with the largest area
property_with_largest_area = property_data.loc[property_data['area_sqft'].idxmax()]
print("3. Property with the largest area:\n", property_with_largest_area)
