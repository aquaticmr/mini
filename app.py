import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model and the scaler
# Ensure these files are in the same directory as app.py
# Load the trained model and the scaler
try:
    # Change the filename here to match your file
    model = pickle.load(open('best_xgboost_model.pkl', 'rb')) 
    
    # This line is correct, so no change is needed
    scaler = pickle.load(open('scaler.pkl', 'rb')) 
except FileNotFoundError:
    print("Error: model.pkl or scaler.pkl not found. Make sure they are in the root directory.")
    exit()


# Define the list of all 55 features the model was trained on, in the correct order
MODEL_FEATURES = [
    'Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction',
    'Job Satisfaction', 'Sleep Duration', 'Work/Study Hours', 'Financial Stress',
    'Gender_Male', 'Profession_Chef', 'Profession_Civil Engineer',
    'Profession_Content Writer', 'Profession_Digital Marketer', 'Profession_Doctor',
    'Profession_Educational Consultant', 'Profession_Entrepreneur', 'Profession_Lawyer',
    'Profession_Manager', 'Profession_Pharmacist', 'Profession_Student',
    'Profession_Teacher', 'Profession_UX/UI Designer', 'Dietary Habits_Moderate',
    'Dietary Habits_Others', 'Dietary Habits_Unhealthy', 'Degree_B.Com', 'Degree_B.Ed',
    'Degree_B.Pharm', 'Degree_B.Tech', 'Degree_BA', 'Degree_BBA', 'Degree_BCA',
    'Degree_BE', 'Degree_BHM', 'Degree_BSc', 'Degree_Class 12', 'Degree_LLB',
    'Degree_LLM', 'Degree_M.Com', 'Degree_M.Ed', 'Degree_M.Pharm', 'Degree_M.Tech',
    'Degree_MA', 'Degree_MBA', 'Degree_MBBS', 'Degree_MCA', 'Degree_MD', 'Degree_ME',
    'Degree_MHM', 'Degree_MSc', 'Degree_Others', 'Degree_PhD',
    'Have you ever had suicidal thoughts ?_Yes', 'Family History of Mental Illness_Yes'
]

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # --- 1. Get User Input ---
    # Create a dictionary from the form data
    form_data = {
        'Age': int(request.form['age']),
        'CGPA': float(request.form['cgpa']),
        'Sleep Duration': float(request.form['sleep_duration']),
        'Work/Study Hours': float(request.form['work_study_hours']),
        'Gender': request.form['gender'],
        'Profession': request.form['profession'],
        'Dietary Habits': request.form['dietary_habits'],
        'Degree': request.form['degree'],
        'Academic Pressure': int(request.form['academic_pressure']),
        'Work Pressure': int(request.form['work_pressure']),
        'Study Satisfaction': int(request.form['study_satisfaction']),
        'Job Satisfaction': int(request.form['job_satisfaction']),
        'Financial Stress': int(request.form['financial_stress']),
        'Suicidal Thoughts': request.form['suicidal_thoughts'],
        'Family History': request.form['family_history']
    }

    # --- 2. Preprocess the Input ---
    # Create a DataFrame with all model features initialized to 0
    input_df = pd.DataFrame(0, index=[0], columns=MODEL_FEATURES)

    # --- 3. Fill the DataFrame with user data ---
    # a) Fill in the numerical features directly
    input_df['Age'] = form_data['Age']
    input_df['CGPA'] = form_data['CGPA']
    input_df['Sleep Duration'] = form_data['Sleep Duration']
    input_df['Work/Study Hours'] = form_data['Work/Study Hours']
    input_df['Academic Pressure'] = form_data['Academic Pressure']
    input_df['Work Pressure'] = form_data['Work Pressure']
    input_df['Study Satisfaction'] = form_data['Study Satisfaction']
    input_df['Job Satisfaction'] = form_data['Job Satisfaction']
    input_df['Financial Stress'] = form_data['Financial Stress']

    # b) Handle One-Hot Encoded features
    # Gender
    if form_data['Gender'] == 'Male':
        input_df['Gender_Male'] = 1

    # Suicidal Thoughts
    if form_data['Suicidal Thoughts'] == 'Yes':
        input_df['Have you ever had suicidal thoughts ?_Yes'] = 1
    
    # Family History
    if form_data['Family History'] == 'Yes':
        input_df['Family History of Mental Illness_Yes'] = 1
    
    # Profession, Degree, Dietary Habits (check if the column exists before setting)
    profession_col = f"Profession_{form_data['Profession']}"
    if profession_col in input_df.columns:
        input_df[profession_col] = 1

    degree_col = f"Degree_{form_data['Degree']}"
    if degree_col in input_df.columns:
        input_df[degree_col] = 1

    diet_col = f"Dietary Habits_{form_data['Dietary Habits']}"
    if diet_col in input_df.columns:
        input_df[diet_col] = 1
        
    # --- 4. Scale the features ---
    try:
        scaled_features = scaler.transform(input_df)
    except Exception as e:
        print(f"Error during scaling: {e}")
        return render_template('index.html', prediction_text='Error during data processing.')

    # --- 5. Make a prediction ---
    prediction = model.predict(scaled_features)
    prediction_proba = model.predict_proba(scaled_features)

    # --- 6. Format the output ---
    prediction_confidence = round(np.max(prediction_proba) * 100, 2)
    if prediction[0] == 1:
        result_text = f"High risk of mental health concerns detected (Confidence: {prediction_confidence}%)"
    else:
        result_text = f"Low risk of mental health concerns detected (Confidence: {prediction_confidence}%)"

    return render_template('index.html', prediction_text=result_text)

# Run the application
if __name__ == "__main__":
    app.run(debug=True)
