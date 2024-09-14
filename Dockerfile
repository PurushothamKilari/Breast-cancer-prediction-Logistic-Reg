
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

RUN python model_generator.py

EXPOSE  5000 
ENV FLASK_APP=app.py

# Run the Flask application when the container launches
CMD ["sh", "-c","python app.py"]