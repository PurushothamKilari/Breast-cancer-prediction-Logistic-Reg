from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from flask_cors import CORS

# Initialize the Flask app
app = Flask(__name__)

CORS(app)
# Load the saved model and scaler
model = joblib.load('breast_cancer_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get input values from the form
        radius_mean = float(request.form.get('radius_mean', 0))
        texture_mean = float(request.form.get('texture_mean', 0))
        perimeter_mean = float(request.form.get('perimeter_mean', 0))
        area_mean = float(request.form.get('area_mean', 0))
        smoothness_mean = float(request.form.get('smoothness_mean', 0))
        compactness_mean = float(request.form.get('compactness_mean', 0))

        # Prepare the input for prediction
        user_data = [[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean]]

        # Scale the input using the scaler
        user_data_scaled = scaler.transform(user_data)

        # Make predictions using the pre-trained model
        prediction = model.predict(user_data_scaled)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
