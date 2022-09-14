#######################
# Single Page Version #
#######################

import streamlit as st
import pandas as pd
import numpy as np
# import pickle as pkl
# from datetime import date, datetime

import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
from sklearn.preprocessing import MinMaxScaler

import os
### Helper Functions and Dictionaries ###

model_inits = {'Linear': LinearRegression(
), 'Random_Forest': RandomForestRegressor()}


def plotly_ts_plot(df, time_range=['1992-01-01', '2019-12-01']):
    """ Returns plotly plot of data over specified time range 
      Args:
        df: (Dataframe) timeseries data with index as datetime type
        time_range: (list of str) two string elements signaling start and end date
      Returns:
        plotly plot object

    """
    df = df.loc[time_range[0]: time_range[1]]
    fig = px.line(df, x=df.index, y=df.columns,
                  template='plotly_dark', markers=True)
    fig.update_layout(title=f'Space Agency Features Plot')
    return fig


def model_training_block(df, select_cols, model_type, test_size=0.2, normalize=True):
    """Return model test set performance plot and errors
        Args:
            df: (Dataframe) timeseries data with index as datetime type
            select_cols: (list[str]) list of strings for columns to use in model
        Returns:
            Fitted Model + Test Peformance of Model plot + two metric measures
    """
    X = df[select_cols]
    y = df['Fuel']

    xtrain, xtest, ytrain, ytest = train_test_split(
        X, y, test_size=test_size, shuffle=False)

    if normalize:
        scaler = MinMaxScaler()
        xtrain = scaler.fit_transform(xtrain)
        xtest = scaler.transform(xtest)

    model = model_inits[model_type]

    model.fit(xtrain, ytrain)

    # Make Predictions
    ytrain_pred = model.predict(xtrain)
    train_mse = mean_squared_error(ytrain_pred, ytrain)
    train_r2 = r2_score(ytrain_pred, ytrain)

    ypred = model.predict(xtest)
    test_mse = mean_squared_error(ytest, ypred)
    test_r2 = r2_score(ytest, ypred)

    x_ax = X.iloc[-len(xtest):].index
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_ax, y=ytest,
                             mode='markers',
                             name='markers'))
    fig.add_trace(go.Scatter(x=x_ax, y=ypred,
                             mode='lines+markers',
                             name='lines+markers'))
    fig.update_layout(title='Model Performance')

    return model, fig, test_mse, test_r2


def model_inference_app(model, feature_values):
    """
    """
    num_features = model.n_features_in_

    return model.predict(np.array(feature_values[:num_features]).reshape(1, -1))[0]


### End Helper Functions ####


### Begin Main Code #####
st.title("Align AI Machine Learning Operations")

st.header("Data Exploration Plots")

# Preprocess data file
data_file = os.path.join('app', 'space_trip.csv')
data = pd.read_csv(data_file, index_col=0)
data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)
data['Time'] = np.arange(len(data.index))

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


### Model Training Block ###

st.header("Model Training")

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

    if st.button('Start Training'):
        fitted_model, figure, metric1, metric2 = model_training_block(
            data, select_cols, model_type, test_size, normalize)
        st.plotly_chart(figure, use_container_width=True)
        m1, m2 = st.columns(2)
        m1.metric("Test MSE", str(np.round(metric1, 3)))
        m2.metric("Test R2", str(np.round(metric2, 3)))


### Model Inference APP ###

st.header("Inference App")

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
