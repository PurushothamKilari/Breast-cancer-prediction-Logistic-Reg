# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app/

# Run the model_generator.py script
RUN python model_generator.py

# Expose port 5000 for the Flask application
EXPOSE  8501 

# Define environment variable for Flask
ENV FLASK_APP=app.py

# Run the Flask application when the container launches
CMD ["sh", "-c", "python -m streamlit run app.py"]