import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
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

def scale_dataset(dataframe, oversample=False):
    if isinstance(dataframe, pd.DataFrame):
        # If dataframe is a Pandas DataFrame
        X = dataframe[dataframe.columns[:-1]].values  # Get features as numpy array
        y = dataframe[dataframe.columns[-1]].values  # Get target as numpy array

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        if oversample:
            ros = RandomOverSampler()
            X, y = ros.fit_resample(X, y)

        # Concatenate X and y into a single array
        data = np.hstack((X, np.reshape(y, (-1, 1))))

        return data, X, y

# Scale and oversample the dataset
data, X_train, y_train = scale_dataset(df, oversample=True)

# Print the lengths of X_train (features) and y_train (target) after preprocessing
print("Gammas:")
print(len(X_train))
print("Hadrons:")
print(len(y_train))
