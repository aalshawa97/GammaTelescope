import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC
import tensorflow as tf

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
    plt.hist(df[df["class"] == 1][label], bins=30, color='blue', label='Gamma', alpha=0.7, density=True)
    plt.hist(df[df["class"] == 0][label], bins=30, color='red', label='Hadron', alpha=0.7, density=True)
    plt.title(f'Histogram of {label}')
    plt.xlabel(label)
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.show()

def scale_dataset(dataframe, oversample=False):
    if isinstance(dataframe, pd.DataFrame):
        # Extract features and target
        X = dataframe[dataframe.columns[:-1]].values  # Get features as numpy array
        y = dataframe[dataframe.columns[-1]].values  # Get target as numpy array

        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        if oversample:
            ros = RandomOverSampler()
            X, y = ros.fit_resample(X, y)

        return X, y

# Scale and oversample the dataset
X_train, y_train = scale_dataset(df, oversample=True)

# Print the lengths of X_train (features) and y_train (target) after preprocessing
print("Training Set Size:")
print(len(X_train))
print("Training Labels Size:")
print(len(y_train))

# Split data into train, validation, and test sets
X_train, y_train = scale_dataset(df, oversample=True)
X_valid, y_valid = scale_dataset(df, oversample=False)
X_test, y_test = scale_dataset(df, oversample=False)

# Create and fit KNN model
knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(X_train, y_train)

# Predict on test data
y_pred_knn = knn_model.predict(X_test)

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

# Train and evaluate SVM model
svm_model = SVC()
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

# Define dropout probability
dropout_prob = 0.5  # Example value, adjust as needed
num_nodes = 32
# Define and compile Neural Network model with Input layer
nm_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),  # Specify input shape here
    tf.keras.layers.Dense(num_nodes, activation='relu'),
    tf.keras.layers.Dropout(dropout_prob),  # Add Dropout layer here
    tf.keras.layers.Dense(num_nodes, activation='relu'),
    tf.keras.layers.Dropout(dropout_prob),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

nm_model.compile(optimizer=tf.keras.optimizers.Adam(),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

nm_model.summary()

# Train the Neural Network model, an epoch is a training cycle
history = nm_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the Neural Network model
loss, accuracy = nm_model.evaluate(X_test, y_test)
print("Neural Network Test Loss:", loss)
print("Neural Network Test Accuracy:", accuracy)
