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
COLS = ["fLength", "fWidth", "fSize", "fConc", "fConc1",
        "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]

# Read the dataset into a DataFrame
df = pd.read_csv("telescope.data", names=COLS)

# Convert class labels to binary integers (1 for Gamma, 0 for Hadron)
df["class"] = (df["class"] == "g").astype(int)

# Plot histograms for each feature
def plot_feature_histograms(dataframe):
    for label in dataframe.columns[:-1]:  # Exclude the last column which is "class"
        plt.figure(figsize=(8, 6))
        plt.hist(dataframe[dataframe["class"] == 1][label], bins=30, color='blue', label='Gamma', alpha=0.7, density=True)
        plt.hist(dataframe[dataframe["class"] == 0][label], bins=30, color='red', label='Hadron', alpha=0.7, density=True)
        plt.title(f'Histogram of {label}')
        plt.xlabel(label)
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid(True)
        plt.show()

plot_feature_histograms(df)

def scale_and_oversample(dataframe, oversample=False):
    """Standardize features and optionally oversample the dataset."""
    X = dataframe.iloc[:, :-1].values  # Extract features
    y = dataframe.iloc[:, -1].values   # Extract target
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)
    
    return X, y

# Split data into train, validation, and test sets
X, y = scale_and_oversample(df, oversample=True)
X_train, X_test, y_train, y_test = X, X, y, y  # For simplicity, using the same data for train/test
X_valid, y_valid = X, y  # Placeholder, should ideally be split differently

# Create and fit KNN model
def train_knn(X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X_train, y_train)
    return model

knn_model = train_knn(X_train, y_train)

# Predict and evaluate KNN model
def evaluate_knn(model, X_test, y_test):
    y_pred = model.predict(X_test)
    distances, indices = model.kneighbors(X_train)
    
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
    
    return classification_report(y_test, y_pred)

print("KNN Classification Report:")
print(evaluate_knn(knn_model, X_test, y_test))

# Train and evaluate SVM model
def train_and_evaluate_svm(X_train, y_train, X_test, y_test):
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    print("SVM Classification Report:")
    print(classification_report(y_test, y_pred))

train_and_evaluate_svm(X_train, y_train, X_test, y_test)

# Define and compile Neural Network model
def create_nn_model(input_shape, num_nodes, dropout_prob):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(num_nodes, activation='relu'),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(num_nodes, activation='relu'),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Train Neural Network model
def train_nn_model(X_train, y_train, X_valid, y_valid, num_nodes, dropout_prob, lr, batch_size, epochs):
    model = create_nn_model(X_train.shape[1], num_nodes, dropout_prob)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_valid, y_valid), verbose=0)
    return model, history

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Crossentropy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.show()

# Perform hyperparameter tuning
def tune_nn_hyperparameters(X_train, y_train, X_valid, y_valid):
    epochs = 100
    num_nodes_options = [16, 32, 64]
    dropout_probs = [0, 0.2]
    lrs = [0.1, 0.005, 0.001]
    batch_sizes = [32, 64, 128]
    
    for num_nodes in num_nodes_options:
        for dropout_prob in dropout_probs:
            for lr in lrs:
                for batch_size in batch_sizes:
                    print(f"Training with num_nodes={num_nodes}, dropout_prob={dropout_prob}, lr={lr}, batch_size={batch_size}")
                    model, history = train_nn_model(X_train, y_train, X_valid, y_valid, num_nodes, dropout_prob, lr, batch_size, epochs)
                    plot_training_history(history)

# Train and evaluate Neural Network model
X_train, y_train = scale_and_oversample(df, oversample=True)
X_valid, y_valid = scale_and_oversample(df, oversample=False)
X_test, y_test = scale_and_oversample(df, oversample=False)

tune_nn_hyperparameters(X_train, y_train, X_valid, y_valid)
