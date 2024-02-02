from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

# Generate a sample dataset (replace this with your actual dataset)
# Assume X contains symptom features and y contains labels (0 or 1)
np.random.seed(42)
X = np.random.rand(100, 5)  # 100 samples, 5 features
y = np.random.randint(2, size=100)  # Binary labels (0 or 1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Get user input for the features of a new patient
new_patient_features = []
for i in range(X.shape[1]):
    feature_value = float(input(f"Enter value for feature {i + 1}: "))
    new_patient_features.append(feature_value)

# Convert the user input to a numpy array and standardize the features
new_patient_features = np.array(new_patient_features).reshape(1, -1)
new_patient_features_scaled = scaler.transform(new_patient_features)

# Get user input for the number of neighbors (k)
k_neighbors = int(input("Enter the number of neighbors (k): "))

# Create a KNN classifier and fit it to the training data
knn_classifier = KNeighborsClassifier(n_neighbors=k_neighbors)
knn_classifier.fit(X_train_scaled, y_train)

# Predict whether the new patient has the medical condition or not
prediction = knn_classifier.predict(new_patient_features_scaled)

# Display the prediction
if prediction[0] == 0:
    print("The model predicts that the patient does not have the medical condition.")
else:
    print("The model predicts that the patient has the medical condition.")
