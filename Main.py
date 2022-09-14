import streamlit as st
import os 
import pandas as pd
import numpy as np
from components.ml_functions import plotly_ts_plot

from components.components import build_header

def run():
    """Runs main page"""
    
    st.set_page_config(
    page_title="MLOps",
    page_icon="🚀",
    )

    st.sidebar.title("Machine Learning Operations")

    build_header("AlignAI MLOps")

    st.header("Data Exploration Plots")

    # Preprocess data file
    data_file = os.path.join('app', 'space_trip.csv')
    data = pd.read_csv(data_file, index_col=0)
    data['Month'] = pd.to_datetime(data['Month'])
    data.set_index('Month', inplace=True)
    data['Time'] = np.arange(len(data.index))
    
    st.session_state['data'] = data

    # Data Visualization
    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Select start date",
            data.index[0],
            min_value=data.index[0],
            max_value=data.index[-1],
        )
    with col2:
        end_date = st.date_input(
            "Select end date",
            data.index[-1],
            min_value=start_date,
            max_value=data.index[-1],
        )

    instructions = """
        Click and drag line chart to select and pan date interval\n
        """

    cols = data.columns.tolist()

    select_cols = st.multiselect(
        "Select Features to compare",
        cols[:-1],
        default=[
            '# of Trips'
        ],
        help=instructions,
    )

    select_cols_df = pd.DataFrame(select_cols).rename(columns={0: "cols"})

    if not select_cols:
        st.stop()

    filtered_df = data[
        select_cols_df["cols"]
    ]

    st.plotly_chart(plotly_ts_plot(filtered_df, time_range=[
                    start_date, end_date]), use_container_width=True)

if __name__ == "__main__":
    run()