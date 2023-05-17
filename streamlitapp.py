import streamlit as st
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

if __name__ == '__main__':

    # st.set_page_config(layout='wide')
    st.title("Income Prediction")
    age = st.number_input('Age', min_value=18, max_value=100)
    workclass = st.selectbox('Work class: ', ('State-gov', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
                                              'Local-gov', 'Private', 'Other'))
    fnlwgt = st.number_input('Final Weight (0 or greater)', min_value=0)

    education = st.selectbox('Education: ', ('PreSchool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th',
                                             'HS Graduation', 'Some college degree', 'Associate VOC', 'Associate acdm', 'Bachelors',
                                             'Masters', 'Professional School', 'Doctorate'))
    education_dict = {'PreSchool':1, '1st-4th':2, '5th-6th':3, '7th-8th':4, '9th':5, '10th':6, '11th':7, '12th':8,
                                             'HS Graduation':9, 'Some college degree':10, 'Associate VOC':11, 'Associate acdm':12, 'Bachelors':13,
                                             'Masters':14, 'Professional School':15, 'Doctorate':16}
    education = education_dict[education]

    marital_status = st.selectbox('Marital Status: ', ('Never Married', 'Married (civilian spouse)', 'Divorced', 'Married (spouse absent)',
                                                       'Separated', 'Widowed', 'Other'))
    marital_status_dict = {'Never Married':'Never-married', 'Married (civilian spouse)':'Married-civ-spouse', 'Divorced':'Divorced',
                           'Married (spouse absent)':'Married-spouse-absent', 'Separated':'Separated', 'Widowed':'Widowed', 'Other':'Other'}
    marital_status = marital_status_dict[marital_status]

    occupation = st.selectbox('Occupation: ', ('Administrative Clerk', 'Executive Manager', 'Professional Speciality',
                                               'Technical Support', 'Sales', 'Handlers - cleaners', 'Transportation & Material moving',
                                               'Farming and / or Fishing', 'Crafting, Repairing', 'Protective Service',
                                               'Machine Operator / Inspector', 'Other Services'))
    occupation_dict = {'Administrative Clerk':'Adm-clerical', 'Executive Manager':'Exec-managerial', 'Professional Speciality':'Prof-specialty',
                                               'Technical Support':'Tech-support', 'Sales':'Sales', 'Handlers - cleaners':'Handlers-cleaners', 'Transportation & Material moving':'Transport-moving',
                                               'Farming and / or Fishing':'Farming-fishing', 'Crafting, Repairing':'Craft-repair', 'Protective Service':'Protective-serv',
                                               'Machine Operator / Inspector':'Machine-op-inspct', 'Other Services':'Other-serv'}
    occupation = occupation_dict[occupation]

    relationship = st.selectbox('Relationship Status: ', ('Not-in-family', 'Husband', 'Wife', 'Unmarried', 'Own-child', 'Other-relative'))

    race = st.selectbox('Race: ', ('White', 'Black', 'Asian-Pac-Islander', 'Other'))

    capital_gain = st.number_input('Capital gain (Could be zero or greater): ', min_value=0)

    capital_loss = st.number_input('Capital loss (Could be zero or greater): ', min_value=0)

    hours_per_week = st.number_input('Hours per week (Could be zero or greater): ', min_value=0)

    native_country = st.selectbox('Native Country: ', ('United-states', 'Mexico', 'Other'))

    gender = st.selectbox('Gender: ', ('Male', 'Female'))
    gender_dict = {'Female':0, 'Male':1}
    gender = gender_dict[gender]
    data = CustomData(age, workclass, fnlwgt, education, marital_status, occupation, relationship,
                      race, capital_gain, capital_loss, hours_per_week, native_country, gender)
    if st.button('Predict'):
        data_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred_result = predict_pipeline.predict(data_df)
        if pred_result == '>50K':
            st.subheader("Estimated Income is greater than 50K")
        else:
            st.subheader("Estimated Income is less than 50K")



