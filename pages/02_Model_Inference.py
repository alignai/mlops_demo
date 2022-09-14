import streamlit as st
import os 
import pandas as pd
import numpy as np
from components.components import build_header
from components.ml_functions import model_training_block, model_inference_app

st.set_page_config(
    page_title="Model Training",
    page_icon="🚀",
    )

st.sidebar.title("Model Inference")

build_header("AlignAI MLOps")


if st.session_state.get('data') is not None:
    data = st.session_state.get('data')
else:
    data_file = os.path.join('app', 'space_trip.csv')
    data = pd.read_csv(data_file, index_col=0)
    data['Month'] = pd.to_datetime(data['Month'])
    data.set_index('Month', inplace=True)
    data['Time'] = np.arange(len(data.index))
    
    st.session_state['data'] = data
    

cols = data.columns.tolist()

### Model Inference APP ###

st.header("Inference App")

instructions = """
    Click and drag line chart to select and pan date interval\n
    """

with st.container():
    col3, col4 = st.columns(2)
    instructions2 = """
        Select features and Model type
        """
    with col3:
        select_cols = st.multiselect(
            "Select Features to Use in the Model",
            cols[:-2],
            default=[
                'Total Passengers'
            ],
            help=instructions,
        )

    with col4:
        model_type = st.selectbox(
            'Select Model Type',
            ('Linear', 'Random_Forest'))

    col5, col6 = st.columns(2)

    with col5:
        normalize = st.selectbox('Normalize?',
                                 (True, False))

    with col6:
        test_size = st.slider('Test Size', 0.0, 1.0, 0.1)

    fitted_model, figure, metric1, metric2 = model_training_block(
        data, select_cols, model_type, test_size, normalize)
    col7, col8 = st.columns(2)

    with col7:
        trips = st.number_input('# of Trips')

    with col8:
        passengers = st.number_input('Total Passengers')

    if st.button('Make Inference'):
        feature_values = [trips, passengers]
        print(feature_values)
        st.metric("Model Inference: ", model_inference_app(
            fitted_model, feature_values))
