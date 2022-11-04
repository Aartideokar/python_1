
import streamlit as st                 #
import pandas as pd
               #
import matplotlib.pyplot as plt


header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()

st.markdown(
    """
    <style>
    .main {
    background-color: #111000;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache
def get_data(filename):
	taxi_data = pd.read_csv(filename)

	return taxi_data


with header:
    st.title('welcome to my awesome data science project!')
    st.text('In this project I look into the transactions of taxis in NYC,....')

with dataset:
    st.header('NYC taxi dataset')
    st.text('I found this dataset on kaggle.com.')


    taxi_data = pd.read_csv('taxi_data.csv')
    st.write(taxi_data.head())

    st.subheader('Pick-up location ID distribution on the NYC dataset.')
    pulocation_dist = pd.DataFrame(taxi_data['PULocationID'].value_counts()).head(50)
    st.bar_chart(pulocation_dist)

    st.subheader('Drop-off location ID distribution on the NYC dataset.')
    dolocation_dist = pd.DataFrame(taxi_data['DOLocationID'].value_counts()).head(50)
    st.bar_chart(dolocation_dist)


with features:
    st.header('The features I created')

    st.markdown('* **first feature:** I created this feature because of this... I calculated it using this logic..')
    st.markdown('* **second feature:** I created this feature because of this... I calculated it using this logic..')


with modelTraining:
    st.header('Here is a list of features in my data:')


    sel_col, disp_col = st.columns(2)


    
    sel_col.write(taxi_data.columns)










