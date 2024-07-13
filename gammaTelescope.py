import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Define the column names
cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1",
        "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]

# Read the dataset into a DataFrame
df = pd.read_csv("telescope.data", names=cols)

# Uncomment the line below to display the first few rows of the DataFrame
# df.head()

# Uncomment the line below to show unique values in the "class"
# df["class"].unique()

# Assuming "class" contains labels like "g" for gamma and "h" for something else
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
#Print the number of gammas and hadrons
print(len(df[df["class"]==1])) #Gamma
print(len(df[df["class"]==0]))#Hadron

