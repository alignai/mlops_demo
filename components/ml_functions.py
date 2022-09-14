import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
from sklearn.preprocessing import MinMaxScaler

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
    model_inits = {'Linear': LinearRegression(
    ), 'Random_Forest': RandomForestRegressor()}
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