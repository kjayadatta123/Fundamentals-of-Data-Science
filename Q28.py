from sklearn.cluster import KMeans
import numpy as np

# Sample dataset (replace this with your actual dataset)
# Assume X contains shopping-related features
np.random.seed(42)
X = np.random.rand(100, 2)  # 100 samples, 2 shopping-related features

# Create a K-Means clustering model
# Create a K-Means clustering model with explicit n_init
kmeans_model = KMeans(n_clusters=3, n_init=10, random_state=42)


# Fit the model to the dataset
kmeans_model.fit(X)

# Get user input for the shopping-related features of a new customer
new_customer_features = []
feature_names = ['Feature 1', 'Feature 2']

for i, feature_name in enumerate(feature_names):
    feature_value = float(input(f"Enter {feature_name} of the new customer: "))
    new_customer_features.append(feature_value)

# Convert the user input to a numpy array
new_customer_features = np.array(new_customer_features).reshape(1, -1)

# Predict the cluster/segment for the new customer
predicted_segment = kmeans_model.predict(new_customer_features)

# Display the predicted segment
print(f"The model predicts that the new customer belongs to segment {predicted_segment[0]}")
