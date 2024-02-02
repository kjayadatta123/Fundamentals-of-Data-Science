from sklearn.linear_model import LinearRegression
import numpy as np

# Sample dataset (replace this with your actual dataset)
# Assume X contains features and y contains prices
np.random.seed(42)
X = np.random.rand(100, 2)  # 100 samples, 2 features (area and number of bedrooms)
y = 50 + 30 * X[:, 0] + 20 * X[:, 1] + np.random.normal(0, 5, 100)  # Linear relationship with some noise

# Create a Linear Regression model
linear_reg_model = LinearRegression()

# Train the model
linear_reg_model.fit(X, y)

# Get user input for the features of a new house
new_house_features = []
feature_names = ['Area', 'Number of Bedrooms']

for i, feature_name in enumerate(feature_names):
    feature_value = float(input(f"Enter {feature_name} of the new house: "))
    new_house_features.append(feature_value)

# Convert the user input to a numpy array
new_house_features = np.array(new_house_features).reshape(1, -1)

# Make a prediction for the price of the new house
predicted_price = linear_reg_model.predict(new_house_features)

# Display the prediction
print(f"The model predicts that the price of the new house is ${predicted_price[0]:,.2f}")
