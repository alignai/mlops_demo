from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import os
import plotly.express as px
import pandas as pd
import numpy as np

try:
    debug = False if os.environ["DASH_DEBUG_MODE"] == "False" else True
except:
    debug = True

app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

server = app.server

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

df = pd.read_csv('space_trip.csv', index_col=0)
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
df['Time'] = np.arange(len(df.index))

app.layout = html.Div(children=[
    html.H1(children='AlignAI MLOps'),
    html.Div(children='''
        Dash: A web application framework for your data.
    '''),
    html.H2(children='Data Exploration Plots'),
    html.Div([
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
    html.Div([
        dcc.Dropdown(
                df.columns[:-1],
                placeholder='Select features to compare',
                id='features-to-compare',
                value=[df.columns[1]],
                multi=True
            ),
    ]),
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
                ['Linear', 'Random Forest'],
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
               id='my-slider'),
        ]),
    ])
])


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

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="8050", debug=debug)