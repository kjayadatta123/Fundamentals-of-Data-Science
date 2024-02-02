import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate synthetic customer data
np.random.seed(42)
num_customers = 200

# Simulate spending patterns
total_amount_spent = np.random.normal(1000, 300, num_customers)
frequency_of_visits = np.random.poisson(10, num_customers)

# Create a DataFrame
data = pd.DataFrame({
    'TotalAmountSpent': total_amount_spent,
    'FrequencyOfVisits': frequency_of_visits
})

# Display the first few rows of the dataset
print(data.head())

# Select features for clustering
features = ['TotalAmountSpent', 'FrequencyOfVisits']
X = data[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow Method
wcss = []  # Within-Cluster-Sum-of-Squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)  # Explicitly set n_init
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.show()

# Based on the Elbow Method, choose the optimal number of clusters
optimal_k = 3  # You can adjust this based on the Elbow Method graph

# Build the K-Means clustering model
kmeans_model = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)  # Explicitly set n_init
clusters = kmeans_model.fit_predict(X_scaled)

# Add the cluster labels to the original dataset
data['Cluster'] = clusters

# Display the resulting clusters and their characteristics
cluster_summary = data.groupby('Cluster')[features].mean()
print(cluster_summary)

# Visualize the clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', edgecolor='k', s=50)
plt.scatter(kmeans_model.cluster_centers_[:, 0], kmeans_model.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids')
plt.title('Customer Segmentation using K-Means Clustering')
plt.xlabel('Total Amount Spent (Standardized)')
plt.ylabel('Frequency of Visits (Standardized)')
plt.legend()
plt.show()
