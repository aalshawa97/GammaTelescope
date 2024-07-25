import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report  
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# Define the column names
cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1",
        "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]

# Read the dataset into a DataFrame
df = pd.read_csv("telescope.data", names=cols)

# Convert class labels to binary integers (1 for Gamma, 0 for Hadron)
df["class"] = (df["class"] == "g").astype(int)

# Plot histograms for each feature
for label in cols[:-1]:  # Exclude the last column which is "class"
    plt.figure(figsize=(6, 4))
    plt.hist(df[df["class"]==1][label], bins=30, color='blue', label='Gamma', alpha=0.7, density=True)
    plt.hist(df[df["class"]==0][label], bins=30, color='red', label='Hadron', alpha=0.7, density=True)
    plt.title(f'Histogram of {label}')
    plt.xlabel(label)
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.show()

def scale_and_oversample(X, y, oversample=False):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)

    return X, y

# Split the dataset into training, validation, and test sets
X = df[df.columns[:-1]].values  # Features
y = df[df.columns[-1]].values  # Target

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scale and oversample the training dataset
X_train, y_train = scale_and_oversample(X_train, y_train, oversample=True)

# Scale the validation and test datasets without oversampling
X_valid, y_valid = scale_and_oversample(X_valid, y_valid, oversample=False)
X_test, y_test = scale_and_oversample(X_test, y_test, oversample=False)

# Create KNN model
knn_model = KNeighborsClassifier(n_neighbors=1)

# Fit the model
knn_model.fit(X_train, y_train)

# Predict on test data
y_pred = knn_model.predict(X_test)

# Predict on training data to get distances to nearest neighbors
distances, indices = knn_model.kneighbors(X_train)

# Plot histogram of distances to nearest neighbor for each class
plt.figure(figsize=(8, 6))
plt.hist(distances[y_train == 1][:, 0], bins=30, color='blue', label='Gamma', alpha=0.7, density=True)
plt.hist(distances[y_train == 0][:, 0], bins=30, color='red', label='Hadron', alpha=0.7, density=True)
plt.title('Histogram of Euclidean Distances to Nearest Neighbor')
plt.xlabel('Distance')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()

print(classification_report(y_test, y_pred))
