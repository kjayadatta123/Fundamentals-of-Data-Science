import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = {
    'CustomerID': [1, 2, 3, 4, 5, 6],
    'AmountSpent': [150, 200, 50, 300, 100, 250],
    'FrequencyOfVisits': [5, 3, 8, 2, 6, 4]
}


transaction_data = pd.DataFrame(data)

X = transaction_data[['AmountSpent', 'FrequencyOfVisits']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

transaction_data['Cluster'] = kmeans.labels_

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, cmap='viridis', marker='o', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('K-Means Clustering of Customers')
plt.xlabel('Amount Spent (Standardized)')
plt.ylabel('Frequency of Visits (Standardized)')
plt.legend()
plt.show()
