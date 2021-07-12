import warnings
warnings.filterwarnings('ignore')
import dash
import dash_table
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
import numpy as np
import pandas as pd
import boto3
from heatwave_binarymodel import HeatwaveBinaryModel
from heatwave_trend_tsmodel import HeatwaveTrendTSModel
import figures as figs

def read_from_s3_bucket(data_object_name):

    s3 = boto3.resource(
        service_name='s3',
        region_name='eu-central-1',
        aws_access_key_id = 'AKIATJJR2V5V27JPS7JA',
        aws_secret_access_key='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'
    )

    data_response = s3.Bucket('s3groupperu').Object(data_object_name).get()['Body']

    return data_response

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df_smp = pd.read_csv(read_from_s3_bucket('data/dim_all_country_info.csv'), index_col=[0,1])

df_country_code = pd.read_csv(
    read_from_s3_bucket('data/dim_all_country_static_info.csv')
    )[['iso2Code','name']].drop_duplicates()

dict_country_code = df_country_code[df_country_code.iso2Code.isin(
    df_smp['country.1'].drop_duplicates().values
)].rename(
    columns = { 'name':'label', 'iso2Code': 'value'}
    ).to_dict(orient='records')

def build_banner():
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.Div(
                id="banner-text",
                children=[
                    html.H5("Heatwave Event Research"),
                    html.H6("Cause and Effect Reporting"),
                ],
            )
        ],
    )

def build_tabs():
    return html.Div(
        id="tabs",
        className="tabs",
        children=[
            dcc.Tabs(
                id="app-tabs",
                value="tab2",
                className="custom-tabs",
                children=[
                    dcc.Tab(
                        id="heatwave-trend-tab",
                        label="Heatwave Trend Spot",
                        value="tab1",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                    dcc.Tab(
                        id="heatwave-binary-model-tab",
                        label="Binary Classification",
                        value="tab2",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    )
                ],
            )
        ],
    )

def build_hw_trend_spot_panel():

    return html.Div(children=[

        html.Div(
            children  = [
                html.H5('Country Selection: '),
                dcc.Dropdown(
                    id = 'country_select',
                    options= dict_country_code,
                    value='BE'
                ),
                html.H5('Seasonal Period (Years): '),
                dcc.Slider(
                    min=2,
                    max=6,
                    value=4,
                    id  = 'seasonal_period',
                    marks = {
                        2: {'label' :'2 yrs'},
                        3: {'label' :'3 yrs'},
                        4: {'label' :'4 yrs'},
                        5: {'label' :'5 yrs'},
                        6:{'label' :'6 yrs'}
                    },
                    included=False
                ),
                html.H5('Begin Year to End Year:'),
                dcc.RangeSlider(
                    id='ts_range',
                    min= 1980,
                    max= 2019,
                    step= 1,
                    value=[1980, 2019],
                    marks = {
                        1980: {'label' :'1980'},
                        1985:{'label' :'1985'},
                        1990: {'label' :'1990'},
                        1995:{'label' :'1995'},
                        2000: {'label' :'2000'},
                        2005:{'label' :'2005'},
                        2010: {'label' :'2010'},
                        2015:{'label' :'2015'},
                        2019:{'label' :'2019'}
                    }
                ),

                dbc.Card(
                    dbc.CardBody('''
                        HWN: HWN yearly number of heat waves
                        HWD: HWD length of the longest yearly event
                        HWF: HWF yearly sum of participating heat waves
                        HWA: HWA hottest day of hottest yearly event
                        HWM: HWM average magnitude of all yearly heat waves
                    '''),
                    className="mb-3",
                )
            ],
            className = 'three columns pretty_container'
        ),

        html.Div(
                children=[
                    html.Div(
                        children=[
                            html.H5("Heatwave indicators over years"),
                            dcc.Graph(id="trend_indicators")
                        ],
                        className="five columns pretty_container"
                    ),
                    html.Div(
                        children=[
                            html.H5("Heatwave scaled indicators over years"),
                            dcc.Graph(id="trend_scaled_indicators")
                        ],
                        className="five columns pretty_container"
                    )
                ]
            ),
        html.Div(
            children = [
                html.Div(
                     dash_table.DataTable(
                        id='table_indcators_container',
                        data= None,
                        columns=[
                            {'id':'Year', 'name': 'Year'},
                            {'id': 'HWN_trend', 'name': 'HWN_trend'},
                            {'id': 'HWF_trend', 'name': 'HWF_trend'},
                            {'id': 'HWD_trend', 'name': 'HWD_trend'},
                            {'id': 'HWA_trend', 'name': 'HWA_trend'},
                            {'id': 'HWM_trend', 'name': 'HWM_trend'}
                        ],
                        style_header= {
                            "backgroundColor": "rgb(2,21,70)",
                            "color": "white",
                            "textAlign": "center",
                        },
                        style_data_conditional=[{"textAlign": "center"}],
                    ),
                    className="five columns pretty_container pkcalc-results-table"
                ),
                html.Div(
                    children = [
                        html.H5("Frequency aspect of Heatwave"),
                        dcc.Graph(id="indicator_details_HWN"),
                    ],
                    className="seven columns pretty_container"
                ),
                html.Div(
                    children = [
                        html.H5("Duration aspect of Heatwave"),
                        dcc.Graph(id="indicator_details_HWF"),
                        dcc.Graph(id="indicator_details_HWD")
                    ],
                    className="four columns pretty_container"
                ),
                html.Div(
                    children = [
                        html.H5("Intensity aspect of Heatwave"),
                        dcc.Graph(id="indicator_details_HWA"),
                        dcc.Graph(id="indicator_details_HWM")
                    ],
                    className="four columns pretty_container"
                )
            ]
        )

    ])

def build_hw_binary_model_panel():

    return html.Div(children =[
        html.Div(
            id = 'selection_card',
            children = [
                html.Div(
                    children = [
                        html.H5('Refit Procedure'),
                        html.Label('Classifier Selection: '),
                        html.Br(),
                        dcc.Dropdown(
                            id = 'classifier_select',
                            options= [
                                {'label': 'Support Vector Classifier', 'value': 'svc'},
                                {'label': 'Random Forest Classififer', 'value': 'rf'}
                            ],
                            value='svc'
                        ),
                        html.Br(),
                        html.Button('Fit the classifier', id='fit_button'),
                        html.Br()
                    ],
                    className = 'pretty_container'
                ),
                html.Div(
                    children = [
                    html.H5('Predicting Procedure'),
                    html.Br(),
                    html.Label('Classifier Selection: '),
                    html.Br(),
                    dcc.Dropdown(
                        id = 'classifier_predict_select',
                        options= [
                            {'label': 'Support Vector Classifier', 'value': 'svc'},
                            {'label': 'Random Forest Classififer', 'value': 'rf'}
                        ],
                        value='svc'
                    ),
                    html.Br(),
                    html.Label('Country Selection: '),
                    dcc.Dropdown(
                        id = 'country_predict_select',
                        options= dict_country_code,
                        value='BE'
                    ),
                    html.Br(),
                    html.Label('Time range of prediction: '),
                    html.Br(),
                    dcc.RangeSlider(
                        id='predict_range',
                        min= 1980,
                        max= 2019,
                        step= 1,
                        value=[1980, 2019],
                        marks = {
                            1980: {'label' :'1980'}, 1985:{'label' :'1985'},
                            1990: {'label' :'1990'}, 1995:{'label' :'1995'},
                            2000: {'label' :'2000'}, 2005:{'label' :'2005'},
                            2010: {'label' :'2010'}, 2015:{'label' :'2015'},
                            2019:{'label' :'2019'}
                        }
                    ),
                    html.Button('Predict', id='predict_button'),
                    html.Br()
                ],
                className = 'pretty_container'
                )
            ],
            className = 'three columns pretty_container'

        ),
        html.Div(
            id = 'result_container',
            children = [
                html.Div(
                    children = [
                        html.H5('Fitting Result:'),
                        html.Br(),
                        html.Div(id='accuracy_output'),
                        html.Div(
                            dash_table.DataTable(
                                id='calssfication_report_table',
                                data= None,
                                columns=[
                                    {'id':'metrics', 'name': 'metrics'},
                                    {'id':'precision', 'name': 'precision'},
                                    {'id': 'recall', 'name': 'recall'},
                                    {'id': 'f1-score', 'name': 'f1-score'},
                                    {'id': 'support', 'name': 'support'}
                                ],
                                style_header= {
                                    "backgroundColor": "rgb(2,21,70)",
                                    "color": "white",
                                    "textAlign": "center",
                                },
                                style_data_conditional=[{"textAlign": "center"}],
                            ),
                            className="six columns pretty_container"
                        ),

                        html.Div(
                            dcc.Graph(id="graph-line-roc-curve"),
                            className = 'five columns pretty_container'
                        )
                    ],
                    className="twelve columns pretty_container"
                ),

                html.Div(
                    children = [
                        html.H5('Predicting Result:'),
                        dash_table.DataTable(
                           id='table_predict_container',
                           data= None,
                           columns= [dict(id=str(i), name=str(i)) for i in np.arange(1980,2020)],
                           style_header= {
                               "backgroundColor": "rgb(2,21,70)",
                               "color": "white",
                               "textAlign": "center",
                           },
                           style_data_conditional=[{"textAlign": "center"}],
                        )
                    ],
                    className="twelve columns pretty_container"
                )
            ],
            className = 'nine columns pretty_container'
        )]
    )


################################################
# app layout
app.layout = html.Div(
    id="big-app-container",
    children=[
        build_banner(),
        html.Div(
            id="app-container",
            children=[
                build_tabs(),
                # Main app
                html.Div(
                    id="app-content",
                    children = [build_hw_trend_spot_panel()]
                ),
            ],
        )
    ],
)
# app callback function

@app.callback(
    [Output("app-content", "children")],
    [Input("app-tabs", "value")]
)
def render_tab_content(tab_switch):

    if tab_switch == 'tab1':
        return [build_hw_trend_spot_panel()]
    if tab_switch == 'tab2':
        return [build_hw_binary_model_panel()]


@app.callback(
    [Output('trend_indicators', 'figure'),
    Output('trend_scaled_indicators', 'figure'),
    Output('table_indcators_container', 'data'),
    Output('indicator_details_HWN', 'figure'),
    Output('indicator_details_HWF', 'figure'),
    Output('indicator_details_HWD', 'figure'),
    Output('indicator_details_HWA', 'figure'),
    Output('indicator_details_HWM', 'figure')],
    [Input('country_select', 'value')],
    Input('seasonal_period', 'value'),
    [Input('ts_range', 'value')]
    )
def update_hw_trend_spot_panel(country_name, seasonal_period, ts_range):

    trd = HeatwaveTrendTSModel()

    df_select = df_smp.loc[country_name]
    df_select = df_select[df_select['year.1'].between(ts_range[0], ts_range[1])]

    trd.refit_procedure(df_select, period = seasonal_period)

    fig_1 = make_subplots(
        rows=5, cols=1,
        shared_xaxes =True, x_title = 'Year',
        )

    fig_1.add_traces(
        [
            go.Bar(y = trd.dataset.HWN_trend, x = trd.dataset.index, name = 'HWN_trend'),
            go.Bar(y = trd.dataset.HWD_trend, x = trd.dataset.index, name = 'HWD_trend'),
            go.Bar(y = trd.dataset.HWF_trend, x = trd.dataset.index, name = 'HWF_trend'),
            go.Bar(y = trd.dataset.HWM_trend, x = trd.dataset.index, name = 'HWM_trend'),
            go.Bar(y = trd.dataset.HWA_trend, x = trd.dataset.index, name = 'HWA_trend')
        ],
        rows=[1, 2, 3, 4, 5],
        cols=[1, 1, 1, 1, 1]
    )

    fig_1.update_layout(
        autosize=True, height=600,
        margin=dict(l=30,r=30,b=5,t=10,pad= 0.06)
    )

    scaled_trend_metrics=  trd.scaled_trend_metrics.stack().reset_index()
    scaled_trend_metrics.columns = ['year','indicator', 'value']

    fig_2 = px.line(scaled_trend_metrics, y = 'value', x= 'year', color = 'indicator')
    fig_2.update_layout(
        autosize=True, height=600,
        margin=dict(l=10,r=10,b=30,t=30)
    )

    tbl = trd.dataset[trd.hw_trend_metrics].apply(lambda r: round(r, 1)) \
        .reset_index(drop= False).rename(columns = {'year':'Year'}).to_dict('records')

    res  = []

    res.append(fig_1)
    res.append(fig_2)
    res.append(tbl)

    for m in trd.hw_metrics:

        detail_metric = \
        trd.dataset[['%s_trend'%m, '%s_trend'%m, '%s_estimated'%m]].stack().reset_index()
        detail_metric.columns = ['year','indicator', 'value']

        fig_m = px.line(detail_metric, y = 'value', x= 'year', color = 'indicator')
        fig_m.update_layout(autosize=True, height=225, margin=dict(l=10,r=10,b=30,t=30))
        res.append(fig_m)

    return res

@app.callback(
    Output('table_predict_container', 'data'),
    [Input('predict_button', 'n_clicks')],
    [   State('classifier_predict_select','value'),
        State('country_predict_select','value'),
        State('predict_range','value'),
    ]
)
def update_hw_predict(n_click, classifier_type, country, time_range):

    hwb = HeatwaveBinaryModel(classifier_type)

    df_select = df_smp.loc[country]
    df_select = df_select[df_select['year.1'].between(time_range[0], time_range[1])]


    df_res = pd.DataFrame(
        index=  np.arange(time_range[0], time_range[1]+1),
        data = {
            'Estimated_Result': hwb.predict_procedure(df_select)
        }
    ).applymap(lambda r: 'Yes' if r else 'No').T

    return df_res.to_dict('records')


@app.callback(
    [Output('calssfication_report_table','data'),
    Output('accuracy_output', component_property='children')],
    Output('graph-line-roc-curve', 'figure'),
    [Input('fit_button', 'n_clicks')],
    [State('classifier_select','value')]
)
def update_hw_bm_refit(n_click, classifier_type):

    hwb = HeatwaveBinaryModel(classifier_type)

    hwb.refit_procedure(df_smp, random_state = 24)

    metrics_table = pd.DataFrame.from_dict(hwb.classification_report).applymap(lambda r: round(r,2)) \
        .T.reset_index().rename(columns = {'index':'metrics'}).to_dict('records')

    roc_figure = figs.serve_roc_curve(model= hwb.model, model_type= hwb.model_type, X_test= hwb.X_test, y_test=hwb.y_test)

    return metrics_table, 'Accuracy Score: ' + str(np.round(hwb.accuracy_score,2)), roc_figure


if __name__ == '__main__':

    app.run_server(debug=False)
