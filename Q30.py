from sklearn.tree import DecisionTreeRegressor, export_text
import pandas as pd

# Sample car dataset (replace this with your actual dataset)
data = {
    'Mileage': [50000, 60000, 70000, 30000, 80000],
    'Age': [3, 4, 2, 5, 1],
    'Brand': ['Toyota', 'Honda', 'Toyota', 'Ford', 'Honda'],
    'EngineType': ['Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol'],
    'Price': [20000, 18000, 22000, 15000, 25000]
}

df = pd.DataFrame(data)

# Get user input for features of the new car
new_car_features = {}
new_car_features['Mileage'] = float(input("Enter the mileage of the new car: "))
new_car_features['Age'] = int(input("Enter the age of the new car: "))
new_car_features['Brand'] = input("Enter the brand of the new car: ")
new_car_features['EngineType'] = input("Enter the engine type of the new car: ")

# Convert user input to a DataFrame for prediction
new_car_df = pd.DataFrame([new_car_features])

# Convert categorical features to numerical using one-hot encoding
df_encoded = pd.get_dummies(df, columns=['Brand', 'EngineType'], drop_first=True)
new_car_df_encoded = pd.get_dummies(new_car_df, columns=['Brand', 'EngineType'], drop_first=True)

# Extract features and target variable from the dataset
X = df_encoded.drop('Price', axis=1)
y = df_encoded['Price']

# Train a Decision Tree Regressor model
cart_model = DecisionTreeRegressor(random_state=42)
cart_model.fit(X, y)

# Make a prediction for the new car
# Ensure that the feature names match those used during training
new_car_df_encoded = new_car_df_encoded.reindex(columns=X.columns, fill_value=0)
predicted_price = cart_model.predict(new_car_df_encoded)

# Display the predicted price
print(f"The predicted price of the new car is ${predicted_price[0]:,.2f}")

# Display the decision path
tree_rules = export_text(cart_model, feature_names=list(X.columns))
print("Decision Path:")
print(tree_rules)
