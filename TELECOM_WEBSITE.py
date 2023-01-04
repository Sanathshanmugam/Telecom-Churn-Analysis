# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 10:29:03 2022

@author: ronit
"""

# Contents of ~/my_app/streamlit_app.py
import streamlit as st

body{
background-image:url('https://i.gifer.com/J4x.gif');
background-position: center;
background-size: cover;
}
def TELECOM_CHURN():

    import numpy as np
    import pickle
    import streamlit as st
    from PIL import Image

    image1 = Image.open('D:/ExcelR/Projetcs/Project with group/from Ronit/For web development/ON STREAMLIT/Images/tech.jpg')

    st.title("Welcome to Our Project")
    st.image(image1)

    st.sidebar.header('User Input Parameters')

    #loading the model
    loaded_model = pickle.load(open('D:/ExcelR/Projetcs/Project with group/from Ronit/For web development/ON STREAMLIT/telecom_churn_trained_model.sav','rb'))
        

    def churn_prediction(input_data):
        
        input_data_1 = np.asarray(input_data)
        #st.write(input_data_1.shape)
        input_data_1_reshaped = input_data_1.reshape(1,-1)
        #st.write(input_data_1_reshaped.shape)
        
        #checking the prediction
        prediction_1 = loaded_model.predict(input_data_1_reshaped)
        
        if(prediction_1[0] == 0):
            return ('This person is not going to churn')
        else:
            return ('This person is going to Churn')
        
        
        
    #def main():
    title_container = st.container()
    col1, col2 = st.columns([1,9])
    image2 = Image.open('D:/ExcelR/Projetcs/Project with group/from Ronit/For web development/ON STREAMLIT/Images/verify.png')
    with title_container:
        with col1:
            st.image(image2,width=60)
        with col2:
            st.title('Telecom Churn Prediction')



    account_length          = st.sidebar.number_input('Account length', min_value = 0)
    str_voice_mail_plan     = st.sidebar.radio('Voice Mail Plan', ['Yes','No'])

    if str_voice_mail_plan == 'Yes':
        voice_mail_plan     = 1
    else:
        voice_mail_plan     = 0
        
    #st.write(voice_mail_plan)

    voice_mail_messages     = st.sidebar.number_input('Voice Mail Messages',min_value=0)
    evening_minutes         = st.sidebar.number_input('Evening Minutes')
    night_minutes           = st.sidebar.number_input('Night Minutes')
    international_minutes   = st.sidebar.number_input('International Minutes')
    customer_service_calls  = st.sidebar.number_input('Customer Service Calls', min_value=0)

    str_international_plan      = st.sidebar.radio('International Plan', ['Yes','No'])

    if str_international_plan == 'Yes':
        international_plan     = 1
    else:
        international_plan     = 0

    #st.write(international_plan)
        
    day_calls               = st.sidebar.number_input('Day Calls', min_value=0)
    evening_calls           = st.sidebar.number_input('Evening Calls', min_value=0)
    night_calls             = st.sidebar.number_input('Night Calls', min_value=0)
    international_calls     = st.sidebar.number_input('International Calls', min_value=0)
    total_charge            = st.sidebar.number_input('Total Charge')

    # code for prediction
    churn_status = ''

    #creating submit button
    if st.button('Predict Churn Status'):
        churn_status= churn_prediction([account_length,voice_mail_plan,voice_mail_messages,evening_minutes,
                                        night_minutes,international_minutes,customer_service_calls,international_plan,
                                        day_calls,evening_calls,night_calls,international_calls,total_charge])


    st.success(churn_status)    



def VISUALISATION():
    st.title("DATA VISUALISATION ")
    st.sidebar.markdown("Page 2 ")

def MODEL_EVALUATION():
    st.title("MODEL EVALUATION")
    st.sidebar.markdown("Page 3 ")

page_names_to_funcs = {
    "Telecom churn": TELECOM_CHURN,
    "Data Visualisation": VISUALISATION,
    "Model Evaluation": MODEL_EVALUATION,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()