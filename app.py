import pandas as pd
import pickle
import streamlit as st
import pandas as pd
import numpy as np

# Trained model and data preprocssing pipeline
model = pickle.load(open('model_train', 'rb'))
data_prep= pickle.load(open("data_transformer", 'rb'))
processed_data = pd.read_csv("defaults_selected.csv")
data_prep.fit(processed_data)
############################################################

def run():
    st.set_page_config(page_title= 'Loan Default', page_icon= 'random', layout='centered', initial_sidebar_state='auto')

    from PIL import Image
    image = Image.open('loan_default.jpg')
    image_deloitte = Image.open('profile.jpg')

    st.image(image)

    st.title("Loan Default Prediction")

    add_selectbox = st.sidebar.selectbox("How would you like to predict?", ("Test","Online","Batch"))

    st.sidebar.info("This app predicts probability of default using Logistic Regression.")
    st.sidebar.success("https://www2.deloitte.com/ch/en.html")

    st.sidebar.image(image_deloitte)

    if add_selectbox == "Test":
        st.header("Please provide Client ID (from 0 to 1599):")
        client_id = st.number_input("Client ID", min_value= 0 , max_value = 1599, value = 2)

        input_df = processed_data.iloc[[client_id]]
        x = data_prep.transform(input_df)
        y = x[0,-1]
        prediction = model.predict([x[0,0:-1]])
        if prediction >= 0.5:
            output1 = 'default'
        else:
            output1 = 'not default'
        if y >= 0.5:
            output2 = 'did'
        else:
            output2 = 'did not' 
        st.success('We predict that Client No.{} will {}, while his/her {}.'.format(client_id, output1, output2))

    if add_selectbox == "Online":

        st.header("Please provide basic information of the client:")

        sex = st.selectbox("Gender", ['Male','Female'])
        age = st.number_input("Age", min_value= 0 , max_value = 100, value = 30)

        if st.checkbox("Married?"):
            marital = "Married"
        else:
            marital = "Never Married"

        credit = st.select_slider('Credit', options = range(0,200000), value = 21)
        
        education = st.selectbox("Education", ['High School', 'University', 'Graduate School'])

        output = ""

        input_dict = {'Credit': credit, 'Gender': sex, 'Education': education, 'Marital': marital, 'Age': age, 'Default': 'No'}
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            x = data_prep.transform(input_df)
            prediction = model.predict([x[0,0:-1]])
            if prediction >= 0.5:
                output = 'default'
            else:
                output = 'not default'
            st.success('We predict that this client will {}.'.format(output))

    if add_selectbox == "Batch":
        file_upload = st.file_uploader("Please upload the .csv file for predictions", type = ["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            X = data_prep.fit_transform(data)[:,0:-1]
            data.insert(6, "Prediction", np.where(model.predict(X) > 0.5, "Yes", "No") )
            st.write(data)

if __name__ == '__main__':
    run()
