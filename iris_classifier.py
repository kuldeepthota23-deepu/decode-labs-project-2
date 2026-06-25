import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# INPUT: Load the dataset [cite: 78]
iris = load_iris()
X = iris.data
y = iris.target

# PROCESS: Train-Test Split [cite: 79]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PROCESS: Feature Scaling [cite: 81]
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PROCESS: Apply KNN Algorithm [cite: 82]
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# OUTPUT: Predictions
predictions = model.predict(X_test)

# OUTPUT: Validation (Confusion Matrix & F1 Score) [cite: 80, 83]
print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, predictions))
print("\n--- Classification Report ---")
print(classification_report(y_test, predictions, target_names=iris.target_names))