from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Sample dataset (replace this with your actual dataset)
# Assume X contains features and y contains labels (0 for not churned, 1 for churned)
np.random.seed(42)
X = np.random.rand(100, 3)  # 100 samples, 3 features
y = np.random.randint(2, size=100)  # Binary labels (0 or 1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression model
logistic_reg_model = LogisticRegression(random_state=42)

# Train the model
logistic_reg_model.fit(X_train, y_train)

# Get user input for the features of a new customer
new_customer_features = []
feature_names = ['Usage Minutes', 'Contract Duration', 'Additional Features']

for i, feature_name in enumerate(feature_names):
    feature_value = float(input(f"Enter {feature_name} of the new customer: "))
    new_customer_features.append(feature_value)

# Convert the user input to a numpy array
new_customer_features = np.array(new_customer_features).reshape(1, -1)

# Make a prediction for whether the new customer will churn or not
prediction = logistic_reg_model.predict(new_customer_features)

# Display the prediction
if prediction[0] == 0:
    print("The model predicts that the new customer will not churn.")
else:
    print("The model predicts that the new customer will churn.")
