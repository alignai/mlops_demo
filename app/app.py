from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# SK Learn
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
from sklearn.preprocessing import MinMaxScaler

try:
    debug = False if os.environ["DASH_DEBUG_MODE"] == "False" else True
except:
    debug = True

app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

server = app.server

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

data_file = os.path.join(__location__, 'space_trip.csv')
df = pd.read_csv(data_file, index_col=0)
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
df['Time'] = np.arange(len(df.index))


# SK Learn 
model_inits = {'Linear': LinearRegression(
), 'Random_Forest': RandomForestRegressor()}

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

app.layout = dbc.Container(html.Div(children=[
    html.H1(children='AlignAI MLOps'),
    html.Div(children='''
        Dash: A web application framework for your data.
    '''),
    html.H2(children='Data Exploration Plots'),
    html.Div([html.P("Select the model date range:"),
            dcc.DatePickerRange(
                id='model-date-range',
                start_date_placeholder_text="Start Date",
                end_date_placeholder_text="End Date",
                calendar_orientation='vertical',
                min_date_allowed=df.index[0],
                max_date_allowed=df.index[-1],
                start_date=df.index[0],
                end_date=df.index[-1]
            ),
        ]),
    html.Div([html.P("Select the features to compare:"),
        dcc.Dropdown(
                df.columns[:-1],
                placeholder='Select features to compare',
                id='features-to-compare',
                value=[df.columns[1]],
                multi=True
            ),
    ]),
    html.Br(),
    html.Div([
        dcc.Graph(id='data-exploration-graph')
    ]),
    html.Br(),
    html.Div([
        html.H2(children="Model Training"),
        html.Div(["Select features to use in the model",
            dcc.Dropdown(
                df.columns[:-1],
                placeholder='Features',
                id='features-to-use',
                value=[df.columns[1]],
                multi=True
            ),
        ]),
        html.Div(["Select Model Type",
            dcc.Dropdown(
                ['Linear', 'Random_Forest'],
                placeholder='Linear',
                id='model-type',
                value=['Linear'],
            ),
        ]),
        html.Div(["Normalize?",
            dcc.Dropdown(
                ['True', 'False'],
                placeholder='True',
                id='normalize-model',
                value=['True'],
            ),
        ]),
        html.Div(["Test Size",
            dcc.Slider(0.1, 1, 0.1,
               value=0.1 ,
               id='test-size'),
        ]),
        html.Button('Submit', id='submit-val', n_clicks=0),
        html.Br(),
        html.Div(id='model-training-output',
             children='Enter a value and press submit')
    ]),
    html.H2(children="Inference App"),
    html.P('Number of Trips:'),
    dcc.Input(id='number-of-trips', type='number', min=0.01, max=0.25, step=0.01),
    html.P('Total Passengers:'),
    dcc.Input(id='total-passengers', type='number', min=0.01, max=0.50, step=0.01),
    html.Br(),
    html.Button('Submit', id='inference-app-submit-val', n_clicks=0),
    html.Br(),
    html.Div(id='inference-app-output',
             children='Enter a value and press submit')
], style={'padding': 10, 'flex': 1}))


@app.callback(
    Output('data-exploration-graph', 'figure'),
    Input('model-date-range', 'start_date'),
    Input('model-date-range', 'end_date'),
    Input('features-to-compare', 'value'))
def update_figure(start_date, end_date, feature_list):
    fig = None
    if start_date and end_date and feature_list:
        tmp_df = df[feature_list]
        tmp_df = tmp_df.loc[start_date: end_date]
        fig = px.line(tmp_df, x=tmp_df.index, y = tmp_df.columns, markers=True)
        fig.update_layout(title=f'Space Agency Features Plot')
    return fig

@app.callback(
    Output('model-training-output', 'children'),
    Input('features-to-use', 'value'),
    Input('model-type', 'value'),
    Input('normalize-model', 'value'),
    Input('test-size', 'value'),
    Input('submit-val', 'n_clicks'),
)
def train_model(features_to_use, model_type, normalize_model, test_size, value):
    if features_to_use and model_type and normalize_model and test_size and value:
        output_text = f"""
        Features to use: {features_to_use}.
        Model type: {model_type}.
        Normalize Model: {normalize_model}.
        Test Size: {test_size}.
        The value is {value}
        """
        fitted_model, figure, metric1, metric2 = model_training_block(
            df, features_to_use, model_type, test_size, normalize_model)
        
        print(metric1)
        print(metric2)
        metric1 = str(np.round(metric1, 3))
        metric2 = str(np.round(metric2, 3))
        return html.Div([
            html.P(f"Test MSE: {metric1}"),
            html.P(f"Test R2: {metric2}"), 
            dcc.Graph(figure=figure)
            ])

@app.callback(
    Output('inference-app-output', 'children'),
    Input('features-to-use', 'value'),
    Input('model-type', 'value'),
    Input('normalize-model', 'value'),
    Input('test-size', 'value'),
    Input('number-of-trips', 'value'),
    Input('total-passengers', 'value'),
    Input('inference-app-submit-val', 'n_clicks'),
)
def model_inference(features_to_use, model_type, normalize_model, test_size, num_trips, total_passengers, value):
    if features_to_use and model_type and normalize_model and test_size and value and num_trips and total_passengers:
        output_text = f"""
        Features to use: {features_to_use}.
        Model type: {model_type}.
        Normalize Model: {normalize_model}.
        Test Size: {test_size}.
        The value is {value}
        Number of trips: {num_trips}
        Total Passengers: {total_passengers}
        """
        fitted_model, figure, metric1, metric2 = model_training_block(
            df, features_to_use, model_type, test_size, normalize_model)
        
        feature_values = [num_trips, total_passengers]
        
        output_value = model_inference_app(
        fitted_model, feature_values)
        # print(metric1)
        # print(metric2)
        # metric1 = str(np.round(metric1, 3))
        # metric2 = str(np.round(metric2, 3))
        return html.Div([
            html.H3(f"Model Inference: {output_value}")
            ])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="8050", debug=debug)