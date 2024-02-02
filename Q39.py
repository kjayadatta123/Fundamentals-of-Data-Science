import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import warnings

# Sample customer transaction data (replace this with your actual dataset)
data = {
    'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'TotalAmountSpent': [100, 200, 150, 300, 250, 120, 180, 220, 200, 250],
    'NumItemsPurchased': [3, 5, 4, 7, 6, 2, 3, 4, 5, 6]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Select relevant features for clustering
features = df[['TotalAmountSpent', 'NumItemsPurchased']]

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply K-Means clustering
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(features_scaled)

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='TotalAmountSpent', y='NumItemsPurchased', hue='Cluster', data=df, palette='viridis')
plt.title('Customer Segmentation using K-Means Clustering')
plt.xlabel('Total Amount Spent')
plt.ylabel('Number of Items Purchased')
plt.show()
