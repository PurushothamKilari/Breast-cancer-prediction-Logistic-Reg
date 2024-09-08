import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the dataset into dataframe
df = pd.read_csv("data.csv")
print(len(df.columns))
print(len(df.index))

# Map "M" to 1 and "B" to 0
df["diagnosis"] = df["diagnosis"].replace({"M": 1, "B": 0})

# Preprocess the data to remove non useful columns
df.info()
df.drop(['id','Unnamed: 32'], axis=1,inplace=True)
print(len(df.columns))

# Making a count vs outcount plot to check the balance in data 
sns.catplot(x="diagnosis", kind="count", data=df, palette="Set2")
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('diagnosis', axis=1), df['diagnosis'], test_size=0.3, random_state=42)

#Scaling the data
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

# Create a linear regression model and fit it to the training data
# model = KNeighborsClassifier(n_neighbors=11)
model =LogisticRegression(C=0.1)
model.fit(X_train, y_train)

# Use the model to make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the accuracy of the model
accuracy=accuracy_score(y_test, y_pred)
print(f"Classification Accuracy:\n {accuracy}")