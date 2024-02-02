import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
# Generate a sample dataset
np.random.seed(42)
num_samples = 200

# Features
age = np.random.randint(20, 80, num_samples)
gender = np.random.choice(['Male', 'Female'], num_samples)
blood_pressure = np.random.randint(90, 150, num_samples)
cholesterol = np.random.randint(120, 300, num_samples)

# Target variable
treatment_outcome = np.random.choice(['Good', 'Bad'], num_samples)

# Create a DataFrame
data = pd.DataFrame({
    'age': age,
    'gender': gender,
    'blood_pressure': blood_pressure,
    'cholesterol': cholesterol,
    'treatment_outcome': treatment_outcome
})

# Display the first few rows of the dataset
print(data.head())

# Encode categorical variables if needed (e.g., using one-hot encoding)
data = pd.get_dummies(data, columns=['gender'])

# Select features and target variable
features = ['age', 'gender_Female', 'gender_Male', 'blood_pressure', 'cholesterol']
target = 'treatment_outcome'

# Prepare the data for training the model
X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a KNN classification model
k_value = 3  # You can adjust the value of k based on your needs
model = KNeighborsClassifier(n_neighbors=k_value)
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='Good')
recall = recall_score(y_test, y_pred, pos_label='Good')
f1 = f1_score(y_test, y_pred, pos_label='Good')
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')
print('\nConfusion Matrix:')
print(conf_matrix)
