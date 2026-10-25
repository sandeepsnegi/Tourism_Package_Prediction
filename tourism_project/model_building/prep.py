import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset directly from the repository data folder
df = pd.read_csv("tourism_project/data/tourism.csv")

# perform data cleaning and remove unnecessary column/s
df.drop(columns=["CustomerID"], inplace=True)

# Clean up some known data issues
df["Gender"].replace("Fe Male", "Female", inplace=True)
df["MaritalStatus"].replace("Unmarried", "Single", inplace=True)

# Prepare the features and target variable
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# split the cleaned dataset into training and testing sets
# stratify=y keeps the (imbalanced) failure ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# save them locally and make the train and test splits available to the next job as a workflow artifact
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Output the message to github action logs.
print("Data prepared: train/test splits written.")
