# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import dill as pickle
from lightgbm import LGBMClassifier
import shap
import matplotlib.pyplot as plt
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px

##################### CONFIGURE SERVER #########################
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
appdash = dash.Dash(__name__, external_stylesheets=external_stylesheets,
  requests_pathname_prefix='/dashboard/')

########################### LAYOUT ############################# 
appdash.layout = html.Div([
    html.Div([
        html.Div(
            dcc.Input(id='input-on-submit',
                      type='number',
                      placeholder='Enter ID client',
                      style={
                          'width': '150px',
                          'height': '40px'
                      })),
            html.Button('Submit',
                        id='submit-val',
                        n_clicks=0,
                        style={
                            'margin-left': '10px',
                            'height': '40px'
                        }),
    ],
             style={
                 'display': 'flex',
                 'margin-left': '600px'
             }),
    html.Div(id='container1',
             style={
                 'margin-top': '10px',
                 'width': '1400px',
                 'height': '50px',
                 'text-align': 'center'
             }),
    html.Div([
        dcc.Graph(id='best2',
                  style={
                      'display': 'inline',
                      'float': 'left',
                      'width': '25%',
                      'height': '300px'
                  }),
        dcc.Graph(id='best1',
                  style={
                      'display': 'inline',
                      'float': 'left',
                      'width': '25%',
                      'height': '300px'
                  }),
        dcc.Graph(id='wors1',
                  style={
                      'display': 'inline',
                      'float': 'left',
                      'width': '25%',
                      'height': '300px'
                  }),
        dcc.Graph(id='wors2',
                  style={
                      'display': 'inline',
                      'float': 'left',
                      'width': '25%',
                      'height': '300px'
                  }),
    ]),
    html.Div([
        html.Div([
            dcc.RadioItems(id='crossfilter-gender-type',
                           options=[{
                               'label': i,
                               'value': i
                           } for i in ['Male', 'Female', 'All']],
                           value='Male',
                           labelStyle={'display': 'inline-block'})
        ],
                 style={
                     'width': '100%',
                     'display': 'inline-block',
                     'margin-left': '35px'
                 }),
        html.Div([
            html.Div([
                html.Div(dcc.Markdown('''Payment rate:'''),
                         style={'margin-left': '35px'}),
                html.Div(dcc.RangeSlider(
                    id='crossfilter-paymentRate--slider',
                    min=0.01,
                    max=0.12,
                    value=[0.01, 0.12],
                    marks={
                        str(pay): str(pay)
                        for pay in np.round(np.linspace(0.01, 0.12, 12), 2)
                    },
                    step=None),
                         style={
                             'width': '100%',
                         }),
            ],
                     style={
                         'width': '33%',
                         'display': 'inline',
                         'float': 'left',
                     }),
            html.Div([
                html.Div(dcc.Markdown('''Own car age:'''),
                         style={'margin-left': '35px'}),
                html.Div(dcc.RangeSlider(
                    id='crossfilter-ownCarAge--slider',
                    min=0,
                    max=60,
                    value=[0, 60],
                    marks={str(year): str(year)
                           for year in range(0, 50, 5)},
                    step=None),
                         style={
                             'width': '100%',
                         }),
            ],
                     style={
                         'width': '33%',
                         'display': 'inline',
                         'float': 'left',
                     }),
            html.Div([
                html.Div(dcc.Markdown('''Age:'''),
                         style={'margin-left': '35px'}),
                html.Div(dcc.RangeSlider(
                    id='crossfilter-daysBirth--slider',
                    min=24,
                    max=68,
                    value=[24, 68],
                    marks={str(year): str(year)
                           for year in range(24, 68, 4)},
                    step=None),
                         style={
                             'width': '100%',
                         }),
            ],
                     style={
                         'width': '33%',
                         'display': 'inline',
                         'float': 'left',
                     }),
        ]),
        html.Div([
            dcc.Graph(id='crossfilter-indicator-scatter1',
                      style={
                          'width': '33%',
                          'display': 'inline',
                          'float': 'left',
                      }),
            dcc.Graph(id='crossfilter-indicator-scatter2',
                      style={
                          'width': '33%',
                          'display': 'inline',
                          'float': 'left',
                      }),
            dcc.Graph(id='crossfilter-indicator-scatter3',
                      style={
                          'width': '33%',
                          'display': 'inline',
                          'float': 'left',
                      })
        ])
    ])
], style={
        'width': '1400px',
        'height': '900px'
})


######################## CALLBACK ###########################
@appdash.callback(
    Output('container1', 'children'),
    Output('best2', 'figure'),
    Output('best1', 'figure'),
    Output('wors1', 'figure'),
    Output('wors2', 'figure'),
    Output('crossfilter-indicator-scatter1', 'figure'),
    Output('crossfilter-indicator-scatter2', 'figure'),
    Output('crossfilter-indicator-scatter3', 'figure'),
    Input('submit-val', 'n_clicks'),
    Input('crossfilter-gender-type', 'value'),
    Input('crossfilter-paymentRate--slider', 'value'),
    Input('crossfilter-ownCarAge--slider', 'value'),
    Input('crossfilter-daysBirth--slider', 'value'),
    State('input-on-submit', 'value')
)
def update_output(n_clicks, gender_type, paymentRate_value,
                  ownCarAge_value, daysBirth_value, value):
    ### Prepare the datas
    clientID = value

    columns_name = pd.read_csv('df.csv', nrows=1).drop(columns='Unnamed: 0')
    df = pd.read_csv('df.csv',
                     nrows=1000,
                     skiprows=clientID)
    df = df.iloc[:, 1:]
    df = pd.DataFrame(data=df.values, columns=columns_name.columns)
    df['OWN_CAR_AGE'] = df['OWN_CAR_AGE'].replace(np.nan, 0)
    df['DAYS_BIRTH'] = df['DAYS_BIRTH'].apply(lambda x: int(x / -365))

    with open('model.pk', 'rb') as f:
        model = pickle.load(f)

    pred_array = model.predict_proba(df,
                                     num_iteration=int(
                                         model.best_iteration_))[:, 1]

    explainer = shap.TreeExplainer(model)
    shap_values_zero, shap_values_one = explainer.shap_values(df)
    shap_expect_values = explainer.expected_value
    df_for_shap = df.copy()

    pred_array_rounded = np.where(pred_array > 0.1, 1., 0.)
    df = pd.concat([df, pd.Series(pred_array_rounded, name='Target')], axis=1)
    df = pd.concat([df, pd.Series(pred_array, name='Predict')], axis=1)

    if pred_array_rounded[0] == 1:
        ifr = """Dear client {}, We are sorry, we can't give you a favorable answer.
        Your banking agent will explain to you our process, it will give you a best understanding of your loan application""".format(clientID)
    elif pred_array_rounded[0] == 0:
        ifr = """Dear client {}, We are happy to announce you that your loan application is accepted.""".format(clientID)

    ### Plot shap force features
    # Prepare datas
    df_client = pd.DataFrame(data=shap_values_one, columns=df_for_shap.columns)

    best_shap = df_client.apply(lambda x: x.nlargest(3).index.tolist(), axis=1)
    wors_shap = df_client.apply(lambda x: x.nsmallest(3).index.tolist(),
                                axis=1)

    best_cli = df.loc[0, best_shap[0]].rename('client')
    best_mean = df.loc[:, best_shap[0]].mean().rename('mean')

    wors_cli = df.loc[0, wors_shap[0]].rename('client')
    wors_mean = df.loc[:, wors_shap[0]].mean().rename('mean')

    best_tar_0 = df.loc[:, best_shap[0]][df['Target'] == 0].mean().rename(
        'paid_loan')
    best_tar_1 = df.loc[:, best_shap[0]][df['Target'] == 1].mean().rename(
        'default')

    wors_tar_0 = df.loc[:, wors_shap[0]][df['Target'] == 0].mean().rename(
        'paid_loan')
    wors_tar_1 = df.loc[:, wors_shap[0]][df['Target'] == 1].mean().rename(
        'default')

    indic_best_shap = pd.concat([best_cli, best_mean, best_tar_0, best_tar_1],
                                axis=1).T
    indic_wors_shap = pd.concat([wors_cli, wors_mean, wors_tar_0, wors_tar_1],
                                axis=1).T

    colors = {
        "client": "goldenrod",
        "mean": "blue",
        "paid_loan": "green",
        "default": "red"
    }
    # Bar 1
    wors_fig1 = px.bar(x=indic_wors_shap.index,
                       y=indic_wors_shap.iloc[:, 0],
                       color=indic_wors_shap.index,
                       color_discrete_map=colors,
                       labels={
                           "y": indic_wors_shap.iloc[:, 0].name,
                           "x": ""
                       })
    # Bar 2
    wors_fig2 = px.bar(x=indic_wors_shap.index,
                       y=indic_wors_shap.iloc[:, 1],
                       color=indic_wors_shap.index,
                       color_discrete_map=colors,
                       labels={
                           "y": indic_wors_shap.iloc[:, 1].name,
                           "x": ""
                       })
    # Bar 3
    best_fig1 = px.bar(x=indic_best_shap.index,
                       y=indic_best_shap.iloc[:, 0],
                       color=indic_best_shap.index,
                       color_discrete_map=colors,
                       labels={
                           "y": indic_best_shap.iloc[:, 0].name,
                           "x": ""
                       })
    # Bar 4
    best_fig2 = px.bar(x=indic_best_shap.index,
                       y=indic_best_shap.iloc[:, 1],
                       color=indic_best_shap.index,
                       color_discrete_map=colors,
                       labels={
                           "y": indic_best_shap.iloc[:, 1].name,
                           "x": ""
                       })

    best_fig1.update_layout(showlegend=False, plot_bgcolor='#F7C1FE')
    best_fig2.update_layout(showlegend=False, plot_bgcolor='#F7C1FE')
    wors_fig1.update_layout(showlegend=False, plot_bgcolor='#C1D6FE ')
    wors_fig2.update_layout(showlegend=False, plot_bgcolor='#C1D6FE ')

    ### Plot scatter filtered
    if gender_type == 'Male':
        dff = df[df['CODE_GENDER'] == 0]
    elif gender_type == 'Female':
        dff = df[df['CODE_GENDER'] == 1]
    elif gender_type == 'All':
        dff = df

    ### Scatter 1
    dff_payment = dff[dff['PAYMENT_RATE'].between(paymentRate_value[0],
                                                  paymentRate_value[1],
                                                  inclusive=True)]

    x = dff_payment['PAYMENT_RATE']
    y = dff_payment['Predict']

    fig1 = px.scatter(dff_payment, x=x, y=y, color="Predict")
    fig1.update_traces(marker_coloraxis=None)

    ### Scatter 2
    dff_car = dff[dff['OWN_CAR_AGE'].between(ownCarAge_value[0],
                                             ownCarAge_value[1],
                                             inclusive=True)]
    x = dff_car['OWN_CAR_AGE']
    y = dff_car['Predict']
    fig2 = px.scatter(dff_car, x=x, y=y, color="Predict")
    fig2.update_traces(marker_coloraxis=None)

    ### Scatter 3
    dff_birth = dff[dff['DAYS_BIRTH'].between(daysBirth_value[0], daysBirth_value[1], inclusive=True)]

    x = dff_birth['DAYS_BIRTH']
    y = dff_birth['Predict']
    fig3 = px.scatter(dff_birth, x=x, y=y, color="Predict")

    return ifr, best_fig2, best_fig1, wors_fig1, wors_fig2, fig1, fig2, fig3

if __name__ == '__main__':
    appdash.run_server(debug=True)