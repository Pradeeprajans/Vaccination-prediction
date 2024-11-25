# H1N1 & Seasonal Flu Vaccination Prediction

This project is a **web-based application** built using **Flask** that predicts the likelihood of an individual getting the **H1N1** or **Seasonal Flu** vaccine based on various personal, behavioral, and demographic factors. It leverages machine learning models trained on vaccination-related data to make predictions. The app collects user input through a web form and uses pre-trained models to predict the likelihood of vaccination.

## Features

- **H1N1 Vaccine Prediction**: Predicts whether an individual is likely to get the H1N1 vaccine based on their responses regarding their concern about H1N1, health behaviors, doctor recommendations, and more.
  
- **Seasonal Flu Vaccine Prediction**: Similarly, predicts the likelihood of receiving the seasonal flu vaccine based on the user's responses.

- **Interactive Web Interface**: The user interacts with a clean, intuitive form where they can select options related to their health, knowledge, behavior, and social interactions.

- **Real-Time Predictions**: Upon submitting the form, the app processes the data through a trained machine learning model and returns the prediction about the likelihood of receiving either the H1N1 or Seasonal Flu vaccine.

## Project Structure

- **app.py**: This is the main Flask application file that runs the web server and handles the routing of user inputs. It includes endpoints for displaying the form, processing form submissions, and serving the prediction results.
  
- **model_h1n1.pkl**: A serialized machine learning model for predicting H1N1 vaccine uptake.
  
- **model_seasonal.pkl**: A serialized machine learning model for predicting Seasonal Flu vaccine uptake.

- **templates/**
  - **index.html**: The HTML template for the main form where users input their details for prediction.
  - **result_h1n1.html**: Displays the result of the H1N1 vaccine prediction.
  - **result_seasonal.html**: Displays the result of the Seasonal Flu vaccine prediction.

- **static/**
  - **style.css**: The CSS file for styling the web pages.

## How the Trained Model Files are Deployed

The machine learning models for both **H1N1 vaccine prediction** and **Seasonal Flu vaccine prediction** are saved as **Pickle (.pkl) files**. These models are loaded into the **Flask backend** when the user submits the form.

### Model Deployment Workflow:
1. **User Submission**: The user submits their responses through a form.
2. **Data Preprocessing**: The Flask application processes the form data to match the format expected by the model.
3. **Model Prediction**:
   - The processed data is passed into the pre-trained models.
   - The model makes a prediction based on the input features.
4. **Result Display**: The Flask app then returns the prediction result to the user through the `result_h1n1.html` or `result_seasonal.html` page.

The models are loaded at the start of the app and are used every time the user submits the form.

## How to Run the Web Application

Follow these steps to run the Flask application locally:

### 1. Clone the Repository
Clone this repository to your local machine:
```bash 
git clone https://github.com/your-username/vaccine-prediction.git
cd vaccine-prediction
```

# 2. Install Dependencies
Ensure you have Python installed. Then, install the necessary dependencies by running:

### 2. Install Dependencies

Ensure you have Python installed. Then, install the necessary dependencies by running:

```bash
pip install -r requirements.txt
```

### 3. Add Model Files

Ensure that the trained model files (`model_h1n1.pkl` and `model_seasonal.pkl`) are placed in the root directory of the project, as they are required for making predictions.

These files contain the pre-trained machine learning models used to predict the likelihood of H1N1 and Seasonal Flu vaccination based on the form data submitted by the user. Without these model files, the application will not function correctly.

### 4. Run the Flask Application

To start the Flask development server, run the following command in your terminal:

```bash
python app.py
```

### 5. Access the Application

After running the Flask application, open your browser and navigate to `http://127.0.0.1:5000`. You should see a form asking for your details. 

Once the form is filled out, you can click either the **"Predict H1N1 Vaccine"** or **"Predict Seasonal Vaccine"** button to receive the vaccination prediction result based on the data you provided.


## Conclusion

This project offers a practical application for predicting vaccination likelihood based on personal, behavioral, and demographic factors. By using machine learning models deployed with **Flask**, users can receive predictions about the **H1N1** and **Seasonal Flu** vaccines in real-time.

The app is designed to be easily deployable, allowing you to run it locally or deploy it on cloud platforms for broader access.
