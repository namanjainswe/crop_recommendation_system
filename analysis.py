import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("dataset/Crop_recommendation.csv")

# Count each crop
crop_counts = data["label"].value_counts()

print(crop_counts)

# Plot crop distribution
plt.figure(figsize=(12, 6))

sns.countplot(
    data=data,
    x="label",
    order=crop_counts.index
)

plt.title("Crop Distribution")
plt.xlabel("Crop")
plt.ylabel("Number of Samples")
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

print("\nStatistical Summary:")
print(data.describe())

# Correlation matrix
correlation = data.drop(columns=["label"]).corr()

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm"
)

plt.title("Feature Correlation")

plt.tight_layout()
plt.show()