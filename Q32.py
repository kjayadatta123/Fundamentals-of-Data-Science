import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample data
house_size = [150, 200, 120, 180, 220, 160, 190]
house_price = [300000, 400000, 250000, 350000, 450000, 320000, 380000]

# Create a DataFrame
data = pd.DataFrame({
    'house_size': house_size,
    'house_price': house_price
})

# Display the data
print(data)

# Scatter plot to visualize the relationship
plt.scatter(data['house_size'], data['house_price'])
plt.title('house_size vs. house_price')
plt.xlabel('house_size')
plt.ylabel('house_price')
plt.show()

# Prepare the data for training the model
X = data[['house_size']]
y = data['house_price']

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

# Visualize the regression line
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.title('house_size vs. house_price (Linear Regression)')
plt.xlabel('house_size')
plt.ylabel('house_price')
plt.show()
