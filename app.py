
import pickle as pkl
import streamlit as st # type: ignore
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,roc_curve
from sklearn.metrics import classification_report
import pandas as pd

# Set page configuration
st.set_page_config(page_title="Diabetes Prediction", layout="wide", page_icon="🧑‍⚕️")

# Load the saved diabetes model
diabetes_model_path = "diabetes-prediction-model.sav"
diabetes_model = pkl.load(open(diabetes_model_path, 'rb'))

# Page title
st.title('Diabetes Prediction using ML')

# Getting the input data from the user
col1, col2, col3 = st.columns(3)

with col1:
    Pregnancies = st.text_input('Number of Pregnancies')

with col2:
    Glucose = st.text_input('Glucose Level')

with col3:
    BloodPressure = st.text_input('Blood Pressure value')

with col1:
    SkinThickness = st.text_input('Skin Thickness value')

with col2:
    Insulin = st.text_input('Insulin Level')

with col3:
    BMI = st.text_input('BMI value')

with col1:
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

with col2:
    Age = st.text_input('Age of the Person')

# Prediction result
diab_diagnosis = ''

# Creating a button for Prediction
if st.button('Diabetes Test Result'):
    try:
        # Convert input to float
        user_input = [float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness), 
                      float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]
        
        # Make prediction
        diab_prediction = diabetes_model.predict([user_input])

        # Display result
        if diab_prediction[0] == 1:
            diab_diagnosis ='The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic ' 

        st.success(diab_diagnosis)
    
    except ValueError:
        st.error("Please enter valid numerical values for all fields.")

 #Accuracy score checking
if st.button('Check Model Accuracy'):
    
    test_data = pd.read_csv("diabetes.csv")
    
    x_test = test_data.drop(columns=["Outcome"])
    y_test = test_data["Outcome"]
    
    y_pred = diabetes_model.predict(x_test)
    
    accuracy = accuracy_score(y_test,y_pred)
    
    st.write(f"Model accuracy on test data: {accuracy *100:.2f}%")
    