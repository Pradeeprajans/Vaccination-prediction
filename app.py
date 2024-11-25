from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

app = Flask(__name__)

# Load the model and preprocessors from pickle files
with open('h1n1_gradient_boosting_model.pkl', 'rb') as f:
    h1n1_model = pickle.load(f)

with open('gradient_boosting_model.pkl', 'rb') as f:
    seasonal_model = pickle.load(f)
    
    
with open('preprocessors.pkl', 'rb') as f:
    preprocessors = pickle.load(f)

# Get the preprocessors
numeric_imputer = preprocessors['numeric_imputer']
scaler = preprocessors['scaler']
categorical_imputer = preprocessors['categorical_imputer']
encoder = preprocessors['encoder']

# Define column categories for preprocessing (adjust based on your dataset)
num_cols = ['h1n1_concern', 'h1n1_knowledge', 'behavioral_antiviral_meds',
            'behavioral_avoidance', 'behavioral_face_mask', 'behavioral_wash_hands',
            'behavioral_large_gatherings', 'behavioral_outside_home', 'behavioral_touch_face',
            'doctor_recc_h1n1', 'doctor_recc_seasonal', 'chronic_med_condition', 'child_under_6_months',
            'health_worker', 'health_insurance', 'opinion_h1n1_vacc_effective', 
            'opinion_h1n1_risk', 'opinion_h1n1_sick_from_vacc', 'opinion_seas_vacc_effective',
            'opinion_seas_risk', 'opinion_seas_sick_from_vacc', 'household_adults', 'household_children']  
ohe_cols = ['age_group', 'education', 'race', 'sex', 'income_poverty', 
            'marital_status', 'rent_or_own', 'employment_status', 'census_msa'] 

@app.route("/", methods=["GET"])
def index():    
    return render_template("index.html")

@app.route("/predict_h1n1", methods=["GET", "POST"])
def predict_h1n1():
    if request.method == "POST":
        # Get input values from the form
        input_data = []
        
        # Numeric inputs
        for col in num_cols:
            input_data.append(request.form[col])  # Adjust as needed for the input names in your form
        
        # Categorical inputs
        for col in ohe_cols:
            input_data.append(request.form[col])  # Adjust as needed for the input names in your form

        # Preprocessing steps

        # Numeric data processing
        input_data_numeric = np.array(input_data[:len(num_cols)]).reshape(1, -1)
        input_data_numeric = numeric_imputer.transform(input_data_numeric)  # This will raise a warning if input is not a DataFrame

        # Convert to DataFrame with correct column names
        input_data_numeric = pd.DataFrame(input_data_numeric, columns=num_cols)

        # Scale the numeric data
        input_data_numeric = scaler.transform(input_data_numeric)
        
        # Convert scaled numeric data to DataFrame
        input_data_numeric = pd.DataFrame(input_data_numeric, columns=num_cols)

        # Categorical data processing
        input_data_categorical = np.array(input_data[len(num_cols):]).reshape(1, -1)
        input_data_categorical = pd.DataFrame(input_data_categorical, columns=ohe_cols)  # Ensure it has column names

        # Impute categorical data
        input_data_categorical = categorical_imputer.transform(input_data_categorical)

        # Encode categorical data
        input_data_categorical = encoder.transform(input_data_categorical)
        
        
        # Convert the encoded categorical data back to a DataFrame
        input_data_categorical = pd.DataFrame(input_data_categorical,columns=encoder.get_feature_names_out(ohe_cols))
        
        
        # Combine the processed numeric and categorical data
        processed_input = pd.concat([input_data_numeric, input_data_categorical], axis=1)
 
        processed_input.columns = [str(col).replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace(' ', '_') for col in processed_input.columns]
        
        # Make prediction
        prediction = h1n1_model.predict(processed_input)
        
        # Render the result
        result = "Yes" if prediction[0] == 1 else "No"
        return render_template("result_h1n1.html", result= result)
    return render_template('index.html')



@app.route('/predict_seasonal', methods=['POST'])
def predict_seasonal():
    if request.method == 'POST':
        # Get input values from the form
        input_data = []
        
        # Numeric inputs
        for col in num_cols:
            input_data.append(request.form[col])  # Adjust as needed for the input names in your form
        
        # Categorical inputs
        for col in ohe_cols:
            input_data.append(request.form[col])  # Adjust as needed for the input names in your form

        # Preprocessing steps

        # Numeric data processing
        input_data_numeric = np.array(input_data[:len(num_cols)]).reshape(1, -1)
        input_data_numeric = numeric_imputer.transform(input_data_numeric)  # This will raise a warning if input is not a DataFrame

        # Convert to DataFrame with correct column names
        input_data_numeric = pd.DataFrame(input_data_numeric, columns=num_cols)

        # Scale the numeric data
        input_data_numeric = scaler.transform(input_data_numeric)
        
        # Convert scaled numeric data to DataFrame
        input_data_numeric = pd.DataFrame(input_data_numeric, columns=num_cols)

        # Categorical data processing
        input_data_categorical = np.array(input_data[len(num_cols):]).reshape(1, -1)
        input_data_categorical = pd.DataFrame(input_data_categorical, columns=ohe_cols)  # Ensure it has column names

        # Impute categorical data
        input_data_categorical = categorical_imputer.transform(input_data_categorical)

        # Encode categorical data
        input_data_categorical = encoder.transform(input_data_categorical)
        
        
        # Convert the encoded categorical data back to a DataFrame
        input_data_categorical = pd.DataFrame(input_data_categorical,columns=encoder.get_feature_names_out(ohe_cols))
        
        
        # Combine the processed numeric and categorical data
        processed_input = pd.concat([input_data_numeric, input_data_categorical], axis=1)

        
        processed_input.columns = [str(col).replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace(' ', '_') for col in processed_input.columns]
        
        # Make prediction
        prediction = seasonal_model.predict(processed_input)
        
        # Render the result
        result = "Yes" if prediction[0] == 1 else "No"
        return render_template("result.html", result= result)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)

