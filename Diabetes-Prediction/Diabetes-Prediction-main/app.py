import pickle
from flask import Flask, render_template, request
import os 

app = Flask(__name__)

# Define the path to your model file
model_filename = 'rf_regressor.pickle'
model_path = os.path.abspath(model_filename)

# Load the machine learning model
with open(model_path, 'rb') as pkl:
    rf_regressor = pickle.load(pkl)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        try:
            # Get input values from the form
            gender = 1 if request.form['Gender'] == 'Male' else 2 if request.form['Gender'] == 'Other' else 0
            smoking_history = 1 if request.form['smoking_history'] == 'never' else 2 if request.form['smoking_history'] == 'former' else 3 if request.form['smoking_history'] == 'current' else 4 if request.form['smoking_history'] == 'not current' else 5 if request.form['smoking_history'] == 'ever' else 0
            Heart_Disease = 1 if request.form['Heart Disease'] == 'Yes' else 0
            Hypertension = 1 if request.form['Hypertension'] == 'Yes' else 0
            Age = float(request.form['Age'])
            BMI = float(request.form['BMI'])
            HbA1c_level = float(request.form['HbA1c_level'])
            blood_glucose_level = float(request.form['blood_glucose_level'])

            # Make prediction using the machine learning model
            prediction = round(rf_regressor.predict([[gender,Age, Hypertension, Heart_Disease,smoking_history,BMI, HbA1c_level,blood_glucose_level]])[0])
        except Exception as e:
            return f"An error occurred: {e}"

    # Render the result in the HTML template
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
