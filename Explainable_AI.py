
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import shap
from lime import lime_tabular

df = pd.read_csv("data.csv")

# Preprocessing
df["diagnosis"] = df["diagnosis"].replace({"M": 1, "B": 0})
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)

# Check data balance
sns.catplot(x="diagnosis", kind="count", data=df, palette="Set2")
plt.show()
#selected features based on the sweetwiz
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

# Create and fit Logistic Regression model
model = LogisticRegression(C=0.1)
model.fit(X_train, y_train)

# SHAP Implementation
explainer_shap = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")
shap_values = explainer_shap.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=selected_features)
shap.force_plot(explainer_shap.expected_value, shap_values[0], X_test[0], feature_names=selected_features)

# LIME Implementation
explainer_lime = lime_tabular.LimeTabularExplainer(X_train, feature_names=selected_features, class_names=['Benign', 'Malignant'], discretize_continuous=True)
i = 0  # Explain the first instance
exp = explainer_lime.explain_instance(X_test[i], model.predict_proba, num_features=6)
exp.show_in_notebook(show_table=True)
exp.save_to_file('lime_explanation.html')
