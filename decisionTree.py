# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load the Iris dataset from sklearn
iris = load_iris()

# Convert the dataset into a DataFrame for easier handling
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]

# Split the data into features (X) and target labels (y)
X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
print(type(X_test))
# Evaluate the model on the test set
y_pred = model.predict(X_test)

plant = [[4.3, 3.0, 1.1, 0.1]]

print(model.predict(plant))

# Plot the decision tree
plt.figure(figsize=(10, 8))
plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True)
plt.show()


print(classification_report(y_test, y_pred))


