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
    # st.write(taxi_data.head())

    st.subheader('Pick-up location ID distribution on the NYC dataset.')
    pulocation_dist = pd.DataFrame(taxi_data['PULocationID'].value_counts()).head(50)
    st.bar_chart(pulocation_dist)

    pulocation_dist = pd.DataFrame(taxi_data['PULocationID'].value_counts()).head(50)
    st.bar_chart(pulocation_dist)


with features:
    st.header('The features I created')

    st.markdown('* **first feature:**')
    st.markdown('* **second feature:**')


with modelTraining:
    st.header('Time to train the model!')
    st.text('Here you get to choose the hyperparameters of the model and see how the performance changes!')

    sel_col, disp_col = st.columns(2)

    max_depth = sel_col, st.slider('What should be the max_depth of the model?', min_value=10, max_value=100, value=20, step=10)

    n_estimators = sel_col.selectbox('How many trees should there be?', options=[100,200,300,'No limit'], index = 0)

    sel_col.text('Here is a list of features in my data:')
    sel_col.write(taxi_data.columns)

    input_feature = sel_col.text_input('Which feature should be used as the input feature?', 'PULocationID')



