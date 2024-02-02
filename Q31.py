import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Load the sample customer dataset
customer_data = pd.read_csv("/Users/lakshminarayanamandi/Downloads/Movies/FODS/data.csv")

# Select relevant features for segmentation
selected_features = ['Age']

# Identify categorical columns
categorical_columns = ['PurchaseHistory', 'BrowsingBehavior']

# Create a column transformer to apply different preprocessing to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), selected_features),
        ('cat', OneHotEncoder(), categorical_columns)
    ])

# Apply the column transformer to the selected features
transformed_data = preprocessor.fit_transform(customer_data[selected_features + categorical_columns])

# Suppress FutureWarning about n_init
with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Determine the optimal number of clusters using the Elbow Method
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # Set n_init explicitly
        kmeans.fit(transformed_data)
        inertia.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()

# Based on the Elbow Method, choose the optimal number of clusters
optimal_k = 3  # Adjust based on the Elbow Method graph

# Apply K-Means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)  # Set n_init explicitly
customer_data['Cluster'] = kmeans.fit_predict(transformed_data)

# Analyze the characteristics of each cluster
cluster_means = customer_data.groupby('Cluster')['Age'].mean()

# Visualize the clusters
sns.pairplot(customer_data, hue='Cluster', palette='Set1', diag_kind='kde')
plt.show()
