import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate a sample dataset
np.random.seed(42)
num_samples = 100

# Features
engine_size = np.random.randint(1000, 4000, num_samples)
horsepower = np.random.randint(80, 400, num_samples)
fuel_efficiency = np.random.uniform(10, 40, num_samples)

# Target variable
car_price = 20000 + 100 * engine_size + 50 * horsepower - 100 * fuel_efficiency + np.random.normal(0, 5000, num_samples)

# Create a DataFrame
data = pd.DataFrame({
    'engine_size': engine_size,
    'horsepower': horsepower,
    'fuel_efficiency': fuel_efficiency,
    'car_price': car_price
})

# Display the first few rows of the dataset
print(data.head())

# Select the features to be used for prediction
selected_features = ['engine_size', 'horsepower', 'fuel_efficiency']

# Scatter plots to visualize the relationships between selected features and car prices
for feature in selected_features:
    plt.scatter(data[feature], data['car_price'])
    plt.title(f'{feature} vs. car_price')
    plt.xlabel(feature)
    plt.ylabel('car_price')
    plt.show()

# Prepare the data for training the model
X = data[selected_features]
y = data['car_price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Provide insights to the marketing team on feature importance
coefficients = pd.DataFrame({'Feature': selected_features, 'Coefficient': model.coef_})
print('\nCoefficients:')
print(coefficients)

# Visualize the regression lines for each selected feature
for i, feature in enumerate(selected_features):
    plt.scatter(X_test[feature], y_test, color='black')
    plt.scatter(X_test[feature], y_pred, color='blue', linewidth=3)
    plt.title(f'{feature} vs. car_price (Linear Regression)')
    plt.xlabel(feature)
    plt.ylabel('car_price')
    plt.show()
