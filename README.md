# Breast Cancer Detection using Logistic Regression
This repository contains code for a machine learning model that uses logistic regression to detect breast cancer based on various features. The model is trained on a dataset of breast cancer cases and uses data scaling to achieve an accuracy of 95.24%.
<br>
### Dataset
The dataset used for this project is the Breast Cancer Wisconsin (Diagnostic) Data Set from Kaggle. It contains 569 samples of malignant and benign tumor cells with 30 features each. The features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.<br>

The dataset can be downloaded from this link: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data <br>

### Dependencies
The code requires the following libraries to run:<br>

1. pandas<br>
2. sklearn<br>
3. seaborn<br>
4. matplotlib<br>
### Usage
To run the code, simply execute the following command in your terminal: <br>

`python model.py`

The code will load the dataset, preprocess it, split it into training and testing sets, scale the data, create and fit a logistic regression model, make predictions on the test set, and evaluate the accuracy of the model.<br>
<br>
You can also use Streamlit to interact with the code and visualize the results. Streamlit is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science.<br>

To run the code with Streamlit, execute the following command in your terminal:<br>

`python app.py`
<br>

You can then access the web app from your browser at http://localhost:5000<br>

### Results
The logistic regression model achieved an accuracy of 95.24% on the test set, which is quite impressive for such a simple model.

