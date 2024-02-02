from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree classifier and train it
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# Get user input for the features of a new flower
new_flower_features = []
feature_names = iris.feature_names

for i, feature_name in enumerate(feature_names):
    feature_value = float(input(f"Enter {feature_name} of the new flower: "))
    new_flower_features.append(feature_value)

# Convert the user input to a numpy array
new_flower_features = np.array(new_flower_features).reshape(1, -1)

# Make a prediction for the new flower
predicted_species = decision_tree.predict(new_flower_features)

# Display the prediction
predicted_species_name = iris.target_names[predicted_species[0]]
print(f"The model predicts that the new flower belongs to the '{predicted_species_name}' species.")
