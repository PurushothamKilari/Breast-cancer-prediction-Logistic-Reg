

import h2o
from flask import Flask, request, jsonify
import pandas as pd
import best_h2o_model
import joblib

# Initialize H2O
h2o.init()

# Load the saved H2O model
model = h2o.load_model('GBM_grid_1_AutoML_1_20240912_145001_model_31')

# Create Flask app
app = Flask(__name__)

# Define a prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    
    scaler = joblib.load('scaler.joblib')

# New data point for prediction (make sure the order of features matches)
    new_data = np.array([[17.99, 10.38, 122.8, 1001, 0.1184, 0.2776]])

# Scale the new data using the previously saved scaler
    new_data_scaled = scaler.transform(new_data)

    # # Get data from request
    # data = request.json

    # # Convert the data to H2OFrame
    # input_data = pd.DataFrame(data)
    # h2o_data = h2o.H2OFrame(new_data_scaled)

    # Predict using the model
    predictions = model.predict(new_data_scaled)


    result = predictions.as_data_frame().to_dict(orient='records')

    return jsonify(result)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
