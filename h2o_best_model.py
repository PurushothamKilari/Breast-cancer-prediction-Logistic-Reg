import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import h2o
from h2o.automl import H2OAutoML

# Initialize H2O
h2o.init()

# Load the dataset into a dataframe
df = pd.read_csv("data.csv")
print(f"Number of columns: {len(df.columns)}")
print(f"Number of rows: {len(df.index)}")

# Map "M" to 1 and "B" to 0
df["diagnosis"] = df["diagnosis"].replace({"M": 1, "B": 0})

# Drop non-useful columns
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
print(f"Number of columns after dropping: {len(df.columns)}")

selected_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean']

X = df[selected_features]
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler (so that we can use the same scaling in future predictions)
joblib.dump(scaler, 'scaler.joblib')

# Convert training data to H2O Frame
train_h2o = h2o.H2OFrame(pd.concat([pd.DataFrame(X_train, columns=selected_features), y_train.reset_index(drop=True)], axis=1))
train_h2o['diagnosis'] = train_h2o['diagnosis'].asfactor()  # Ensure the target is treated as categorical

x = selected_features  # Features
y = 'diagnosis'  

aml = H2OAutoML(max_runtime_secs=300, seed=7654321, balance_classes=True)

# Train AutoML
aml.train(x=x, y=y, training_frame=train_h2o)

# Display the leaderboard
lb = aml.leaderboard
print(lb.head())

# Save the best model
best_model = aml.leader
h2o.save_model(best_model, path="best_h2o_model")

# Shutdown H2O
# h2o.shutdown(prompt=False)
