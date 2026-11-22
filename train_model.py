import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Load dataset
data = pd.read_csv("dataset/Crop_recommendation.csv")

# Input features
X = data[
    [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall"
    ]
]

# Target
y = data["label"]

print("Input Features:")
print(X.head())

print("\nTarget:")
print(y.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

from sklearn.tree import DecisionTreeClassifier

# Create Decision Tree
dt_model = DecisionTreeClassifier(
    random_state=42
)

# Train model
dt_model.fit(X_train, y_train)

print("Decision Tree training completed!")

from sklearn.metrics import accuracy_score

# Make predictions
dt_prediction = dt_model.predict(X_test)

# Calculate accuracy
dt_accuracy = accuracy_score(
    y_test,
    dt_prediction
)

print("Decision Tree Accuracy:", dt_accuracy)

from sklearn.ensemble import RandomForestClassifier

# Create Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
rf_model.fit(X_train, y_train)

print("Random Forest training completed!")

# Make predictions
rf_prediction = rf_model.predict(X_test)

# Calculate accuracy
rf_accuracy = accuracy_score(
    y_test,
    rf_prediction
)

print("Random Forest Accuracy:", rf_accuracy)

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print("Decision Tree :", dt_accuracy)
print("Random Forest :", rf_accuracy)

import matplotlib.pyplot as plt

models = [
    "Decision Tree",
    "Random Forest"
]

accuracies = [
    dt_accuracy,
    rf_accuracy
]

plt.figure(figsize=(8, 5))

plt.bar(models, accuracies)

plt.ylim(0, 1)

plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")

plt.show()

from sklearn.metrics import classification_report

print("\nRandom Forest Classification Report:")
print(
    classification_report(
        y_test,
        rf_prediction
    )
)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_test,
    rf_prediction
)

plt.figure(figsize=(12, 10))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=rf_model.classes_,
    yticklabels=rf_model.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")

plt.tight_layout()
plt.show()

importance = rf_model.feature_importances_

features = X.columns

plt.figure(figsize=(10, 5))

plt.bar(
    features,
    importance
)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

import joblib

# Save Decision Tree
joblib.dump(
    dt_model,
    "models/decision_tree.pkl"
)

# Save Random Forest
joblib.dump(
    rf_model,
    "models/random_forest.pkl"
)

print("\nModels saved successfully!")

sample = [[
    90,      # N
    42,      # P
    43,      # K
    20.8,    # Temperature
    82.0,    # Humidity
    6.5,     # pH
    202.9    # Rainfall
]]

prediction = rf_model.predict(sample)

print("\nRecommended Crop:", prediction[0])