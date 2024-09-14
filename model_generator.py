import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset into a dataframe
df = pd.read_csv("data.csv")
print(f"Number of columns: {len(df.columns)}")
print(f"Number of rows: {len(df.index)}")

# Map "M" to 1 and "B" to 0
df["diagnosis"] = df["diagnosis"].replace({"M": 1, "B": 0})

# Drop non-useful columns
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
print(f"Number of columns after dropping: {len(df.columns)}")

# Making a count plot to check the balance in data
sns.catplot(x="diagnosis", kind="count", data=df, palette="Set2")
plt.show()

# Select 6 features to use (you can adjust these as needed)
selected_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean']

# Split the data into features and target
X = df[selected_features]
y = df['diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a Logistic Regression model and fit it to the training data
model = LogisticRegression(C=0.1)
model.fit(X_train, y_train)

# Save the model to a file using joblib
joblib.dump(model, 'breast_cancer_model.joblib')

# Save the scaler as well (so that we can use the same scaling in future predictions)
joblib.dump(scaler, 'scaler.joblib')

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Classification Accuracy: {accuracy}")
