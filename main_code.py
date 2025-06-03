import base64
import io
import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import tushare as ts
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import Callback
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import dashscope
import random
import json
matplotlib.use('Agg')  # 使用非交互式后端避免线程问题


os.environ['OMP_NUM_THREADS'] = '1'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 初始化Tushare Pro
TOKEN = "自己的token"
ts.set_token(TOKEN)
pro = ts.pro_api()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "股票行情分析预测系统"

# 创建菜单栏组件
def create_navbar():
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("数据展示", id="main-link", href="#", active=True)),
            dbc.NavItem(dbc.NavLink("聚类分析", id="cluster-link", href="#")),
            dbc.NavItem(dbc.NavLink("决策树分析", id="tree-link", href="#")),
            dbc.NavItem(dbc.NavLink("SVM预测", id="svm-link", href="#")),
            dbc.NavItem(dbc.NavLink("LSTM预测", id="lstm-link", href="#")),  # 新增LSTM预测链接
            dbc.NavItem(dbc.NavLink("股票云图", id="cloud-link", href="#")),
            dbc.NavItem(dbc.NavLink("新闻事件", id="news-link", href="#")),
        ],
        brand="股票分析预测系统 v1.4",  # 更新版本号
        brand_href="#",
        color="primary",
        dark=True,
        className="mb-4"
    )

# 创建主行情卡片
def build_main_card():
    return dbc.Card([
        dbc.CardBody([
            html.Label("股票代码（支持多个 使用逗号分隔）"),
            dcc.Input(
                id='stock-code',
                type='text',
                value='000001.SZ,000002.SZ,600000.SH',
                className="mb-3"
            ),
            html.Label("日期范围选择"),
            dcc.DatePickerRange(
                id='date-range',
                start_date='2023-01-01',
                end_date=pd.to_datetime('today').strftime('%Y-%m-%d'),
                display_format='YYYY-MM-DD',
                className="mb-3"
            ),
            html.Label("K线类型"),
            dcc.Dropdown(
                id='k-type',
                options=[
                    {'label': '日K', 'value': 'D'},
                    {'label': '周K', 'value': 'W'},
                    {'label': '月K', 'value': 'M'}
                ],
                value='D',
                className="mb-3"
            ),
            html.Label("指标显示"),
            dcc.Dropdown(
                id='indicators',
                options=[
                    {'label': 'MACD', 'value': 'macd'},
                    {'label': 'KDJ', 'value': 'kdj'},
                    {'label': 'RSI', 'value': 'rsi'},
                    {'label': '布林带', 'value': 'boll'},
                    {'label': '均线', 'value': 'ma'}
                ],
                value=['macd', 'ma'],
                multi=True,
                className="mb-3"
            ),
            dbc.Button("查询", id='submit-btn', color="primary", className="w-100")
        ])
    ])

# 创建聚类分析卡片
def build_cluster_card():
    return dbc.Card([
        dbc.CardBody([
            html.Label("聚类数目 (K)"),
            dcc.Input(
                id='k-value', 
                type='number',
                min=2, max=10, step=1,
                value=2,
                className="mb-3"
            ),
            html.Div("聚类数范围: 2-10", style={'font-size': '0.8rem', 'color': 'gray'}),
            dbc.Button("执行聚类", id='cluster-btn', color="primary", className="w-100 mt-3")
        ])
    ])

# 创建决策树卡片
def build_decision_tree_card():
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Label("分析周期（年）"),
                dcc.Input(
                    id='years-slider',
                    type='number',
                    min=1, max=5, step=1,
                    value=1,
                    className="w-100 mb-3"
                )
            ], className="mb-3"),
            html.Div([
                html.Label("涨跌阈值(%)"),
                dcc.Input(id='threshold', type='number', value=20, className="w-100 mb-3")
            ], className="mb-3"),
            html.Div([
                html.Label("决策树深度"),
                dcc.Input(
                    id='tree-depth',
                    type='number',
                    min=2, max=10, step=1,
                    value=2,
                    className="w-100 mb-3"
                )
            ], className="mb-3"),
            html.Div([
                html.Label("行业平均PE"),
                dcc.Input(id='industry-pe', type='number', value=15, className="w-100 mb-3")
            ], className="mb-3"),
            html.Div([
                html.Label("PB警戒阈值"),
                dcc.Input(id='pb-threshold', type='number', value=2, className="w-100")
            ], className="mb-3"),
            dbc.Button("执行分析", id='analyze-btn', color="primary", className="w-100 mt-3")
        ])
    ])

# 创建svm模块
def build_svm_card():
    return dbc.Card([
        dbc.CardBody([
            #html.Div("使用主数据展示的股票代码和日期范围", className="mb-3", style={'color': 'gray'}),
            html.Div([
                html.Label("训练集比例 (%)"),
                dcc.Slider(
                    id='train-ratio',
                    min=60,
                    max=90,
                    step=5,
                    value=80,
                    marks={i: f'{i}%' for i in range(60, 91, 10)}
                )
            ], className="mb-3"),
            html.Div([
                html.Label("核函数"),
                dcc.Dropdown(
                    id='kernel-type',
                    options=[
                        {'label': '线性核', 'value': 'linear'},
                        {'label': '多项式核', 'value': 'poly'},
                        {'label': '径向基核', 'value': 'rbf'},
                        {'label': 'Sigmoid核', 'value': 'sigmoid'}
                    ],
                    value='rbf',
                    clearable=False
                )
            ], className="mb-3"),
            dbc.Button("执行预测", id='svm-btn', color="primary", className="w-100 mt-3")
        ])
    ])

# 创建LSTM预测参数卡片
def build_lstm_card():
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Label("预测天数"),
                dcc.Input(
                    id='forecast-days',
                    type='number',
                    min=1,
                    max=30,
                    step=1,
                    value=3,
                    className="w-100 mb-3"
                )
            ], className="mb-3"),
            html.Div([
                html.Label("回看窗口大小"),
                dcc.Slider(
                    id='window-size',
                    min=10,
                    max=60,
                    step=5,
                    value=30,
                    marks={i: str(i) for i in range(10, 61, 10)}
                )
            ], className="mb-3"),
            html.Div([
                html.Label("LSTM层数"),
                dcc.Dropdown(
                    id='lstm-layers',
                    options=[
                        {'label': '1层', 'value': 1},
                        {'label': '2层', 'value': 2},
                        {'label': '3层', 'value': 3}
                    ],
                    value=2,
                    clearable=False
                )
            ], className="mb-3"),
            html.Div([
                html.Label("训练轮次"),
                dcc.Slider(
                    id='epochs',
                    min=20,
                    max=100,
                    step=10,
                    value=30,
                    marks={i: str(i) for i in range(20, 101, 20)}
                )
            ], className="mb-3"),
            dbc.Button("执行预测", id='lstm-btn', color="primary", className="w-100 mt-3")
        ])
    ])


# 2. 创建云图模块布局（简化版，只保留云图）
def build_cloud_card():
    return dbc.Card([
        dbc.CardBody([
            html.Label("板块选择"),
            dcc.Dropdown(
                id='board-selector',
                options=[
                    {'label': '上证主板', 'value': 'sh'},
                    {'label': '深证主板', 'value': 'sz'},
                    {'label': '创业板', 'value': 'cyb'},
                    {'label': '科创板', 'value': 'kcb'},
                    {'label': '北交所', 'value': 'bj'}
                ],
                value=['sh', 'sz', 'cyb'], 
                multi=True,
                className="mb-3"
            ),
            html.Label("股票数量 (每板块)"),
            dcc.Slider(
                id='stock-count',
                min=5,
                max=30,
                step=5,
                value=10,
                marks={i: str(i) for i in range(5, 31, 5)}
            ),
            html.Div("根据市值和流动性选择代表性股票", style={'font-size': '0.8rem', 'color': 'gray'}),
            dbc.Button("生成云图", id='cloud-btn', color="primary", className="w-100 mt-3")
        ])
    ])

# 创建新闻事件分析卡片
def build_news_card():
    return dbc.Card([
        dbc.CardBody([
            html.Label("股票/公司名称"),
            dcc.Input(
                id='news-keyword',
                type='text',
                placeholder='输入股票代码或公司名称',
                className="mb-3 w-100"
            ),
            html.Label("时间范围"),
            dcc.DatePickerRange(
                id='news-date-range',
                start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                end_date=datetime.now().strftime('%Y-%m-%d'),
                display_format='YYYY-MM-DD',
                className="mb-3"
            ),
            html.Label("情感过滤"),
            dcc.Dropdown(
                id='sentiment-filter',
                options=[
                    {'label': '全部', 'value': 'all'},
                    {'label': '积极', 'value': 'positive'},
                    {'label': '中性', 'value': 'neutral'},
                    {'label': '消极', 'value': 'negative'}
                ],
                value='all',
                className="mb-3"
            ),
            html.Label("影响力阈值"),
            dcc.Slider(
                id='impact-slider',
                min=1,
                max=10,
                step=1,
                value=5,
                marks={i: str(i) for i in range(1, 11)}
            ),
            dbc.Button("分析新闻", id='analyze-news-btn', color="primary", className="w-100 mt-3")
        ])
    ])

# 添加一个存储当前活动标签的状态
app.layout = html.Div([
    dcc.Store(id='active-tab', data='main-tab'),
    
    create_navbar(),
    
    # 主行情模块
    dbc.Container(id='main-module', children=[
        dbc.Row([
            dbc.Col(build_main_card(), md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Tabs(id="stock-tabs", value='tab-0', className="mb-3"),
                        # 这里使用charts-container
                        dcc.Loading(id="loading-charts", type="default", children=html.Div(id="charts-container")),
                        html.H4("每日涨跌情况", className="mt-4"),
                        dash_table.DataTable(
                            id='stock-table',
                            columns=[],
                            data=[],
                            style_table={'overflowX': 'auto'},
                            page_size=10
                        ),
                        dcc.Loading(id="loading", type="default", children=html.Div(id="loading-output"))
                    ])
                ])
            ], md=9)
        ])
    ]),
    
    # 聚类分析模块 
    dbc.Container(id='cluster-module', style={'display': 'none'}, children=[
        dbc.Row(dbc.Col(html.H2("股票聚类分析", className="text-center my-4"))),
        dbc.Row([
            dbc.Col(build_cluster_card(), md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(id='cluster-plot'),
                        dcc.Graph(id='custom-index-chart'),
                        html.H4("聚类结果"),
                        dash_table.DataTable(id='cluster-table', columns=[], data=[], page_size=10)
                    ])
                ])
            ], md=9)
        ])
    ]),
    
    # 决策树模块 
    dbc.Container(id='tree-module', style={'display': 'none'}, children=[
        dbc.Row(dbc.Col(html.H2("价值投资决策分析", className="text-center my-4"))),
        dbc.Row([
            dbc.Col(build_decision_tree_card(), md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div(id='tree-graph'),
                        dcc.Graph(id='feature-importance'),
                        html.H4("分类结果", className="mt-4"),
                        dash_table.DataTable(id='classification-table', columns=[], data=[], page_size=5),
                        html.H4("投资建议", className="mt-4"),
                        dash_table.DataTable(
                            id='investment-advice',
                            columns=[
                                {'name': '代码', 'id': 'code'},
                                {'name': '预测', 'id': 'prediction'},
                                {'name': 'PE', 'id': 'pe'},
                                {'name': '行业PE', 'id': 'industry_pe'},
                                {'name': 'PB', 'id': 'pb'},
                                {'name': '建议', 'id': 'advice'},
                                {'name': '置信度', 'id': 'confidence'}
                            ],
                            style_data_conditional=[
                                {'if': {'column_id': 'advice', 'filter_query': '{advice} = "买入"'},
                                 'backgroundColor': 'lightgreen'},
                                {'if': {'column_id': 'advice', 'filter_query': '{advice} = "卖出"'},
                                 'backgroundColor': 'lightcoral'}
                            ]
                        )
                    ])
                ])
            ], md=9)
        ])
    ]),

    # 新增SVM预测模块
    dbc.Container(id='svm-module', style={'display': 'none'}, children=[
        dbc.Row(dbc.Col(html.H2("SVM股票预测", className="text-center my-4"))),
        dbc.Row([
            dbc.Col(build_svm_card(), md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(id='svm-performance-chart'),
                        html.H5("模型混淆矩阵", className="mt-4"),
                        dcc.Graph(id='svm-confusion-matrix'),  # 新增混淆矩阵图表
                        html.H4("测试集预测结果", className="mt-4"),
                        dash_table.DataTable(
                            id='svm-results-table',
                            columns=[],
                            data=[],
                            style_table={'overflowX': 'auto'},
                            page_size=10
                        ),
                        html.Div(id='svm-accuracy', className="mt-3"),
                        html.H4("投资建议", className="mt-4"),
                        dash_table.DataTable(
                            id='svm-advice-table',
                            columns=[
                                {'name': '代码', 'id': 'code'},
                                {'name': '预测涨跌', 'id': 'prediction'},
                                {'name': '置信度', 'id': 'confidence'},
                                {'name': '建议', 'id': 'advice'}
                            ],
                            style_data_conditional=[
                                {'if': {'column_id': 'advice', 'filter_query': '{advice} = "买入"'},
                                'backgroundColor': 'lightgreen'},
                                {'if': {'column_id': 'advice', 'filter_query': '{advice} = "观望"'},
                                'backgroundColor': 'lightyellow'},
                                {'if': {'column_id': 'advice', 'filter_query': '{advice} = "卖出"'},
                                'backgroundColor': 'lightcoral'}
                            ]
                        )
                    ])
                ])
            ], md=9)
        ])
    ]),


    # 新增LSTM预测模块
    dbc.Container(id='lstm-module', style={'display': 'none'}, children=[
        dbc.Row(dbc.Col(html.H2("LSTM股票预测", className="text-center my-4"))),
        dbc.Row([
            dbc.Col(build_lstm_card(), md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(id='lstm-forecast-chart'),
                        html.Div("灰色区域表示预测区间", 
                                 style={'color': 'gray', 'font-size': '0.8rem', 'text-align': 'center'}),
                        html.H4("预测结果详情", className="mt-4"),
                        dash_table.DataTable(
                            id='lstm-results-table',
                            columns=[],
                            data=[],
                            style_table={'overflowX': 'auto'},
                            page_size=10
                        ),
                        html.H4("投资建议", className="mt-4"),
                        dash_table.DataTable(
                            id='lstm-advice-table',
                            columns=[
                                {'name': '代码', 'id': 'code'},
                                {'name': '预测涨幅(%)', 'id': 'predicted_change'},
                                {'name': '置信度', 'id': 'confidence'},
                                {'name': '建议', 'id': 'advice'}
                            ],
                            style_data_conditional=[
                                {'if': {'column_id': 'advice', 'filter_query': '{advice} = "强烈买入"'},
                                 'backgroundColor': '#00cc96'},
                                {'if': {'column_id': 'advice', 'filter_query': '{advice} = "买入"'},
                                 'backgroundColor': 'lightgreen'},
                                {'if': {'column_id': 'advice', 'filter_query': '{advice} = "持有"'},
                                 'backgroundColor': 'lightyellow'},
                                {'if': {'column_id': 'advice', 'filter_query': '{advice} = "卖出"'},
                                 'backgroundColor': 'lightcoral'},
                                {'if': {'column_id': 'advice', 'filter_query': '{advice} = "强烈卖出"'},
                                 'backgroundColor': '#ff6699'}
                            ]
                        )
                    ])
                ])
            ], md=9)
        ])
    ]),
    # 新增云图模块
    dbc.Container(id='cloud-module', style={'display': 'none'}, children=[
        dbc.Row(dbc.Col(html.H2("股票云图分析", className="text-center my-4"))),
        dbc.Row([
            dbc.Col(build_cloud_card(), md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(id='cloud-plot'),
                        html.Div("点代表股票，大小表示成交量，颜色表示涨跌幅度", 
                                 style={'color': 'gray', 'font-size': '0.9rem', 'text-align': 'center'})
                    ])
                ], className="h-100")
            ], md=9)
        ])
    ]),

    # 新增新闻事件模块
    dbc.Container(id='news-module', style={'display': 'none'}, children=[
        dbc.Row(dbc.Col(html.H2("股票新闻事件分析", className="text-center my-4"))),
        dbc.Row([
            dbc.Col(build_news_card(), md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-news",
                            type="circle",
                            children=[
                                dbc.Row([
                                    dbc.Col([
                                        html.H4("新闻情感趋势", className="mb-3"),
                                        dcc.Graph(id='sentiment-trend-chart')
                                    ], md=6),
                                    dbc.Col([
                                        html.H4("重大事件日历", className="mb-3"),
                                        dash_table.DataTable(
                                            id='event-calendar',
                                            columns=[
                                                {'name': '日期', 'id': 'date'},
                                                {'name': '事件类型', 'id': 'event_type'},
                                                {'name': '影响力', 'id': 'impact'}
                                            ],
                                            style_cell={'textAlign': 'center'},
                                            style_header={'fontWeight': 'bold'},
                                            style_data_conditional=[
                                                {
                                                    'if': {'column_id': 'impact', 'filter_query': '{impact} > 7'},
                                                    'backgroundColor': 'rgba(255, 0, 0, 0.1)',
                                                    'fontWeight': 'bold'
                                                }
                                            ],
                                            page_size=5
                                        )
                                    ], md=6)
                                ]),
                                html.H4("相关新闻摘要", className="mt-4"),
                                html.Div(id='news-summary', style={
                                    'maxHeight': '400px',
                                    'overflowY': 'scroll',
                                    'border': '1px solid #eee',
                                    'padding': '10px',
                                    'borderRadius': '5px'
                                })
                            ]
                        )
                    ])
                ])
            ], md=9)
        ])
    ])
], className="mt-4")

# 更新切换标签回调
@app.callback(
    [Output('main-module', 'style'),
     Output('cluster-module', 'style'),
     Output('tree-module', 'style'),
     Output('svm-module', 'style'),
     Output('lstm-module', 'style'),  # 新增LSTM模块
     Output('cloud-module', 'style'),
     Output('news-module', 'style'),
     Output('main-link', 'active'),
     Output('cluster-link', 'active'),
     Output('tree-link', 'active'),
     Output('svm-link', 'active'),
     Output('lstm-link', 'active'),  # 新增LSTM链接
     Output('cloud-link', 'active'),
     Output('news-link', 'active'),
     Output('active-tab', 'data')],
    [Input('main-link', 'n_clicks'),
     Input('cluster-link', 'n_clicks'),
     Input('tree-link', 'n_clicks'),
     Input('svm-link', 'n_clicks'),
     Input('lstm-link', 'n_clicks'),  # 新增LSTM链接
     Input('cloud-link', 'n_clicks'),
     Input('news-link', 'n_clicks')],
    [State('active-tab', 'data')]
)
def switch_tab(main_clicks, cluster_clicks, tree_clicks, svm_clicks, lstm_clicks, cloud_clicks, news_clicks, active_tab):
    ctx = callback_context
    if not ctx.triggered:
        button_id = 'main-link'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # 设置活动标签
    if button_id == 'main-link':
        return {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, True, False, False, False, False, False, False, 'main-tab'
    elif button_id == 'cluster-link':
        return {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, False, True, False, False, False, False, False, 'cluster-tab'
    elif button_id == 'tree-link':
        return {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, False, False, True, False, False, False, False, 'tree-tab'
    elif button_id == 'svm-link':
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, False, False, False, True, False, False, False, 'svm-tab'
    elif button_id == 'lstm-link':  # 新增LSTM标签
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, False, False, False, False, True, False, False, 'lstm-tab'
    elif button_id == 'cloud-link':
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, False, False, False, False, False, True, False, 'cloud-tab'
    elif button_id == 'news-link':
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, False, False, False, False, False, False, True, 'news-tab'

# =========================================
# 数据展示-工具函数 
# =========================================
def calculate_ma(df, periods=[5, 10, 20, 60]):
    """计算移动平均线"""
    for period in periods:
        df[f'MA{period}'] = df['close'].rolling(window=period).mean()
    return df

def calculate_macd(df):
    """计算MACD指标"""
    if len(df) < 26:
        return pd.Series(), pd.Series(), pd.Series()
    df = df.sort_values('trade_date')
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist

def calculate_kdj(df, n=9, m1=3, m2=3):
    """计算KDJ指标"""
    if len(df) < n:
        return pd.Series(), pd.Series(), pd.Series()
    
    low_list = df['low'].rolling(window=n).min()
    high_list = df['high'].rolling(window=n).max()
    rsv = (df['close'] - low_list) / (high_list - low_list) * 100
    
    k = rsv.ewm(alpha=1/m1).mean()
    d = k.ewm(alpha=1/m2).mean()
    j = 3 * k - 2 * d
    
    return k, d, j

def calculate_rsi(df, period=14):
    """计算RSI指标"""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_boll(df, period=20):
    """计算布林带指标"""
    sma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return sma, upper, lower

def fetch_stock_data(stock_codes, start_date, end_date, k_type='D'):
    """获取股票数据，支持不同K线类型"""
    dfs = []
    for code in stock_codes:
        try:
            if '.' not in code:
                print(f"无效代码格式: {code}")
                continue
                
            # 根据K线类型选择不同的数据接口
            if k_type == 'D':
                df = pro.daily(
                    ts_code=code,
                    start_date=start_date,
                    end_date=end_date,
                    fields="ts_code,trade_date,open,high,low,close,vol,pct_chg"
                )
            elif k_type == 'W':
                df = pro.weekly(
                    ts_code=code,
                    start_date=start_date,
                    end_date=end_date,
                    fields="ts_code,trade_date,open,high,low,close,vol,pct_chg"
                )
            elif k_type == 'M':
                df = pro.monthly(
                    ts_code=code,
                    start_date=start_date,
                    end_date=end_date,
                    fields="ts_code,trade_date,open,high,low,close,vol,pct_chg"
                )
            
            if df.empty:
                print(f"无数据: {code}")
                continue
            if len(df) < 10:  # 确保有足够数据点
                print(f"数据不足: {code} 只有{len(df)}条记录")
                continue
                
            df['code'] = code
            dfs.append(df)
        except Exception as e:
            print(f"获取 {code} 失败: {str(e)}")
    
    if dfs:
        combined_df = pd.concat(dfs)
        combined_df['trade_date'] = pd.to_datetime(combined_df['trade_date'], format='%Y%m%d')
        return combined_df.sort_values(['code', 'trade_date'])
    return pd.DataFrame()

def create_candlestick_chart(df, code, indicators):
    """创建K线图和技术指标图"""
    df = df.sort_values('trade_date')
    df = calculate_ma(df)  # 计算移动平均线
    
    # 创建K线图
    candlestick = go.Candlestick(
        x=df['trade_date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='K线'
    )
    
    # 添加均线
    ma_traces = []
    for period in [5, 10, 20, 60]:
        if f'MA{period}' in df.columns:
            ma_traces.append(go.Scatter(
                x=df['trade_date'],
                y=df[f'MA{period}'],
                mode='lines',
                name=f'{period}日均线',
                line=dict(width=1.5)
            ))
    
    # 创建布林带
    boll_traces = []
    if 'boll' in indicators:
        sma, upper, lower = calculate_boll(df)
        boll_traces = [
            go.Scatter(x=df['trade_date'], y=upper, name='布林上轨', line=dict(color='rgba(0,128,0,0.5)', width=1)),
            go.Scatter(x=df['trade_date'], y=sma, name='布林中轨', line=dict(color='rgba(128,128,128,0.5)', width=1)),
            go.Scatter(x=df['trade_date'], y=lower, name='布林下轨', fill='tonexty', 
                      line=dict(color='rgba(255,0,0,0.5)', width=1), fillcolor='rgba(255,0,0,0.1)')
        ]
    
    # 创建K线图布局
    layout = go.Layout(
        title=f'{code} K线图',
        xaxis=dict(title='日期', rangeslider=dict(visible=False)),
        yaxis=dict(title='价格'),
        showlegend=True,
        height=400
    )
    
    fig_candle = go.Figure(data=[candlestick] + ma_traces + boll_traces, layout=layout)
    
    # 创建成交量图
    volume = go.Bar(
        x=df['trade_date'],
        y=df['vol'],
        name='成交量',
        marker_color='rgba(100,149,237,0.7)'
    )
    
    # 添加成交量均线
    vol_ma_traces = []
    for period in [5, 10]:
        vol_ma = df['vol'].rolling(window=period).mean()
        vol_ma_traces.append(go.Scatter(
            x=df['trade_date'],
            y=vol_ma,
            mode='lines',
            name=f'{period}日成交量均线',
            line=dict(width=1.5, color='orange')
        ))
    
    fig_volume = go.Figure(data=[volume] + vol_ma_traces, layout=dict(
        title='成交量',
        xaxis=dict(title='日期'),
        yaxis=dict(title='成交量'),
        height=400,
        showlegend=True
    ))
    
    # 创建技术指标图
    indicator_figs = []
    
    # MACD指标
    if 'macd' in indicators:
        macd_line, macd_signal, macd_hist = calculate_macd(df)
        if not macd_line.empty:
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=df['trade_date'], y=macd_line, name='MACD线'))
            fig_macd.add_trace(go.Scatter(x=df['trade_date'], y=macd_signal, name='信号线', line_dash='dot'))
            fig_macd.add_trace(go.Bar(x=df['trade_date'], y=macd_hist, name='柱状图'))
            fig_macd.update_layout(title="MACD指标", height=400)
            indicator_figs.append(dcc.Graph(figure=fig_macd))
    
    # KDJ指标
    if 'kdj' in indicators:
        k, d, j = calculate_kdj(df)
        if not k.empty:
            fig_kdj = go.Figure()
            fig_kdj.add_trace(go.Scatter(x=df['trade_date'], y=k, name='K值'))
            fig_kdj.add_trace(go.Scatter(x=df['trade_date'], y=d, name='D值'))
            fig_kdj.add_trace(go.Scatter(x=df['trade_date'], y=j, name='J值'))
            fig_kdj.update_layout(title="KDJ指标", height=400)
            indicator_figs.append(dcc.Graph(figure=fig_kdj))
    
    # RSI指标
    if 'rsi' in indicators:
        rsi = calculate_rsi(df)
        if not rsi.empty:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df['trade_date'], y=rsi, name='RSI'))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            fig_rsi.update_layout(title="RSI指标", height=400)
            indicator_figs.append(dcc.Graph(figure=fig_rsi))
    
    return [
        dcc.Graph(figure=fig_candle),
        dcc.Graph(figure=fig_volume),
        *indicator_figs
    ]

# =========================================
# 云图展示-工具函数
# =========================================

# 获取板块股票
def get_board_stocks(board, count=10):
    """获取指定板块的代表性股票"""
    # 定义各板块的示例股票代码（实际应用中应从API获取）
    board_stocks = {
    'sh': [
        '600000.SH', '600036.SH', '600519.SH', '601318.SH', '601398.SH', 
        '601288.SH', '601857.SH', '601988.SH', '601628.SH', '601766.SH',
        '600276.SH', '600171.SH', '600601.SH', '600406.SH', '600436.SH',
        '600438.SH', '600415.SH', '600887.SH', '600690.SH', '601012.SH',
        '601088.SH', '601601.SH', '603019.SH', '603166.SH', '603711.SH',
        '603057.SH', '605199.SH', '601811.SH', '601939.SH', '601328.SH'
    ],
    'sz': [
        '000001.SZ', '000002.SZ', '000333.SZ', '000651.SZ', '000858.SZ',
        '002415.SZ', '002475.SZ', '002594.SZ', '002607.SZ', '002714.SZ',
        '000063.SZ', '000100.SZ', '000157.SZ', '000338.SZ', '000423.SZ',
        '000538.SZ', '000568.SZ', '000625.SZ', '000725.SZ', '000768.SZ',
        '000776.SZ', '000895.SZ', '000932.SZ', '000983.SZ', '001696.SZ',
        '001965.SZ', '002001.SZ', '002024.SZ', '002142.SZ', '002352.SZ'
    ],
    'cyb': [
        '300059.SZ', '300122.SZ', '300124.SZ', '300142.SZ', '300144.SZ',
        '300146.SZ', '300251.SZ', '300347.SZ', '300498.SZ', '300750.SZ',
        '300308.SZ', '300760.SZ', '300015.SZ', '300024.SZ', '300033.SZ',
        '300058.SZ', '300136.SZ', '300274.SZ', '300316.SZ', '300363.SZ',
        '300413.SZ', '300454.SZ', '300496.SZ', '300595.SZ', '301536.SZ',
        '301590.SZ', '301302.SZ', '301368.SZ', '301429.SZ', '301408.SZ'  
    ],
    'kcb': [
        '688008.SH', '688019.SH', '688036.SH', '688111.SH', '688116.SH',
        '688126.SH', '688169.SH', '688185.SH', '688256.SH', '688981.SH',
        '688200.SH', '688202.SH', '688207.SH', '688212.SH', '688223.SH',
        '688298.SH', '688363.SH', '688390.SH', '688516.SH', '688599.SH',
        '688636.SH', '688382.SH', '688049.SH', '688772.SH', '688252.SH',
        '688609.SH', '688777.SH', '688668.SH', '688551.SH', '688696.SH'  
    ],
    'bj': [
        
        '920799.BJ', '920819.BJ', '920445.BJ', '920489.BJ', '920167.BJ',  
        '920682.BJ', '430047.BJ', '430090.BJ', '830779.BJ', '830809.BJ',
        '831010.BJ', '831152.BJ', '832000.BJ', '832566.BJ', '837748.BJ',
        '835185.BJ', '835368.BJ', '836077.BJ', '836239.BJ', '837242.BJ',
        '838030.BJ', '838171.BJ', '838275.BJ', '838670.BJ', '838924.BJ',
        '839725.BJ', '871642.BJ', '871981.BJ', '873167.BJ', '873339.BJ'
    ]
}
    
    # 返回指定数量的股票
    return board_stocks.get(board, [])[:count]


# =========================================
# 新闻情感分析函数（使用API）
# =========================================

# 配置API Key
os.environ['DASHSCOPE_API_KEY'] = 'your_api_key_here'  # 替换为你的API Key

def analyze_news_sentiment(news_text):
    """调用百炼API分析新闻情感"""
    try:
        response = dashscope.Generation.call(
            model='qwen-plus',
            prompt=f"作为金融分析师，请判断以下新闻对相关股票的影响（输出JSON格式）："
                   f"{news_text}\n"
                   "输出格式：{'sentiment': 'positive/negative/neutral', 'confidence': 0-1, 'impact_score': 1-10, 'keywords': '关键词1,关键词2'}"
        )
        
        if response.status_code == 200:
            # 尝试解析AI生成的JSON
            result_text = response.output['text'].strip()
            if result_text.startswith('{') and result_text.endswith('}'):
                analysis = json.loads(result_text)
                return {
                    'sentiment': analysis.get('sentiment', 'neutral'),
                    'confidence': float(analysis.get('confidence', 0.5)),
                    'impact_score': int(analysis.get('impact_score', 5)),
                    'keywords': analysis.get('keywords', '')
                }
            else:
                # 处理非JSON响应
                return {'error': 'API返回格式错误'}
        else:
            return {'error': f'API调用失败: {response.message}'}
    except Exception as e:
        return {'error': f'解析失败: {str(e)}'}

# 获取模拟新闻数据
def get_news_data(keyword, start_date, end_date):
    """模拟获取新闻数据"""
    news_items = []
    date_range = pd.date_range(start_date, end_date)
    
    sentiments = ['positive', 'neutral', 'negative']
    topics = ['财报发布', '新产品上市', '行业政策', '并购重组', '高管变动', '监管处罚']
    
    for i in range(15):  # 生成15条模拟新闻
        date = random.choice(date_range).strftime('%Y-%m-%d')
        sentiment = random.choice(sentiments)
        topic = random.choice(topics)
        
        # 根据情感生成不同内容
        if sentiment == 'positive':
            content = f"{keyword}发布重要公告：{topic}取得重大进展，市场预期乐观"
        elif sentiment == 'negative':
            content = f"{keyword}遭遇不利因素：{topic}影响公司前景，投资者需警惕风险"
        else:
            content = f"{keyword}相关动态：{topic}情况通报，市场反应中性"
        
        news_items.append({
            'date': date,
            'title': f"{keyword}{topic}相关新闻",
            'content': content,
            'source': random.choice(['新浪财经', '东方财富', '证券时报', '华尔街见闻'])
        })
    
    return news_items


# =========================================
# SVM预测工具函数
# =========================================


def prepare_features(df):
    """准备SVM模型的特征"""
    # 计算技术指标
    df = df.sort_values('trade_date')
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['close'].rolling(5).std()
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma_diff'] = df['ma5'] - df['ma20']
    df['rsi'] = calculate_rsi(df)
    
    # 创建目标变量（未来5天涨跌）
    df['target'] = (df['close'].shift(-5) > df['close']).astype(int)
    
    # 删除缺失值
    df = df.dropna()
    
    return df

def train_svm_model(df, kernel='rbf', train_ratio=0.8):
    """训练SVM模型并返回预测结果"""
    # 选择特征
    features = ['returns', 'volatility', 'ma5', 'ma10', 'ma20', 'ma_diff', 'rsi', 'vol']
    X = df[features]
    y = df['target']
    
    # 标准化特征
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, train_size=train_ratio, shuffle=False
    )
    
    # 训练SVM模型
    model = SVC(kernel=kernel, probability=True, random_state=42)
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # 创建结果DataFrame
    results = pd.DataFrame({
        'Date': df.iloc[-len(y_test):]['trade_date'],
        'Actual': y_test,
        'Predicted': y_pred,
        'Probability': y_prob
    })
    
    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    return results, accuracy, conf_matrix, class_report, model

def generate_advice(df, model, latest_data):
    """生成投资建议"""
    # 准备特征
    features = ['returns', 'volatility', 'ma5', 'ma10', 'ma20', 'ma_diff', 'rsi', 'vol']
    X = df[features].tail(1)
    
    # 预测
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]
    
    # 生成建议
    if prediction == 1 and probability > 0.7:
        advice = "买入"
    elif prediction == 1 and probability > 0.5:
        advice = "观望"
    elif prediction == 0 and probability > 0.7:
        advice = "卖出"
    else:
        advice = "持有"
    
    return prediction, probability, advice


# =========================================
# 新增LSTM预测函数
# =========================================

def lstm_predict(stock_data, forecast_days=3, window_size=30, lstm_layers=2, epochs=30):
    """
    LSTM股票价格预测函数
    """
    # 只使用收盘价
    df = stock_data[['trade_date', 'close']].copy()
    df.set_index('trade_date', inplace=True)
    
    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df.values)
    
    # 创建训练数据集
    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # 构建LSTM模型
    model = Sequential()
    
    # 添加LSTM层
    for i in range(lstm_layers):
        return_sequences = True if i < lstm_layers - 1 else False
        if i == 0:
            model.add(LSTM(units=50, return_sequences=return_sequences, input_shape=(X.shape[1], 1)))
        else:
            model.add(LSTM(units=50, return_sequences=return_sequences))
        model.add(Dropout(0.2))
    
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # 训练模型
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
    
    # 预测未来价格
    inputs = df[-window_size:].values
    inputs = scaler.transform(inputs)
    
    future_predictions = []
    confidence_scores = []
    
    for _ in range(forecast_days):
        x_input = inputs[-window_size:]
        x_input = np.reshape(x_input, (1, window_size, 1))
        
        # 预测下一天的价格
        prediction = model.predict(x_input, verbose=0)
        
        # 计算置信度（基于预测值与最近价格的波动）
        recent_std = np.std(df['close'][-5:])
        confidence = max(0, min(1, 1 - abs(prediction[0][0] - scaled_data[-1][0]) / (recent_std + 1e-7)))
        
        # 保存预测结果和置信度
        future_predictions.append(prediction[0][0])
        confidence_scores.append(confidence)
        
        # 更新输入序列
        inputs = np.append(inputs, prediction)
        inputs = inputs[-window_size:]
    
    # 反归一化预测结果
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)
    
    # 创建预测日期
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
    
    # 准备结果数据
    results = []
    for i, date in enumerate(future_dates):
        results.append({
            'date': date.strftime('%Y-%m-%d'),
            'predicted_price': future_predictions[i][0],
            'confidence': confidence_scores[i]
        })
    
    return results


# =========================================
# 主回调函数 
# =========================================
@app.callback(
    [Output('stock-tabs', 'children'),
     Output('charts-container', 'children'),
     Output('stock-table', 'data'),
     Output('stock-table', 'columns')],
    [Input('submit-btn', 'n_clicks')],
    [State('stock-code', 'value'),
     State('date-range', 'start_date'),
     State('date-range', 'end_date'),
     State('k-type', 'value'),
     State('indicators', 'value')]
)
def update_main_charts(n_clicks, stock_code, start_date, end_date, k_type, indicators):
    if not n_clicks:
        return [], [], [], []
    
    start_date_fmt = start_date.replace("-", "")
    end_date_fmt = end_date.replace("-", "")
    stock_codes = [c.strip().upper() for c in stock_code.split(',') if c.strip()]
    
    try:
        # 获取数据
        df = fetch_stock_data(stock_codes, start_date_fmt, end_date_fmt, k_type)
        if df.empty:
            return [], [], [], []
        
        # 创建股票标签页
        tabs = []
        for i, code in enumerate(df['code'].unique()):
            tabs.append(dcc.Tab(label=code, value=f'tab-{i}'))
        
        # 生成数据表
        table_df = df.rename(columns={
            'ts_code': '代码', 'trade_date': '日期', 'open': '开盘价',
            'high': '最高价', 'low': '最低价', 'close': '收盘价',
            'vol': '成交量', 'pct_chg': '涨跌幅%'
        })
        
        # 返回标签页和默认内容
        return [
            tabs,
            html.Div(id='charts-content'),
            table_df.to_dict('records'),
            [{'name': col, 'id': col} for col in table_df.columns]
        ]
    except Exception as e:
        print(f"主图表错误: {str(e)}")
        return [], [], [], []

# =========================================
# 标签页切换回调
# =========================================
@app.callback(
    Output('charts-content', 'children'),
    [Input('stock-tabs', 'value'),
     Input('stock-code', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('k-type', 'value'),
     Input('indicators', 'value')]
)
def update_tab_content(active_tab, stock_code, start_date, end_date, k_type, indicators):
    # 解析活动标签页索引
    if not active_tab:
        return []
    
    idx = int(active_tab.split('-')[-1])
    stock_codes = [c.strip().upper() for c in stock_code.split(',') if c.strip()]
    
    if idx >= len(stock_codes):
        return []
    
    code = stock_codes[idx]
    
    try:
        # 获取当前股票数据
        start_date_fmt = start_date.replace("-", "")
        end_date_fmt = end_date.replace("-", "")
        df = fetch_stock_data([code], start_date_fmt, end_date_fmt, k_type)
        
        if df.empty:
            return html.Div(f"没有找到 {code} 的数据", className="text-center mt-5")
        
        # 创建图表
        return create_candlestick_chart(df, code, indicators)
    except Exception as e:
        print(f"更新标签页内容错误: {str(e)}")
        return html.Div("图表生成错误", className="text-center mt-5")

# =========================================
# 聚类分析回调（保持不变）
# =========================================
@app.callback(
    [Output('cluster-plot', 'figure'),
     Output('custom-index-chart', 'figure'),
     Output('cluster-table', 'data'),
     Output('cluster-table', 'columns')],
    [Input('cluster-btn', 'n_clicks')],
    [State('stock-code', 'value'),
     State('date-range', 'start_date'),
     State('date-range', 'end_date'),
     State('k-value', 'value')]
)
def update_clusters(n_clicks, stock_code, start_date, end_date, k):
    if not n_clicks or k is None:
        return [go.Figure(), go.Figure(), [], []]

    # 确保聚类数在有效范围内
    k = int(k)
    if k < 2:
        k = 2
    elif k > 10:
        k = 10

    stock_codes = [c.strip().upper() for c in stock_code.split(',') if c.strip()]
    start_date = start_date.replace("-", "")
    end_date = end_date.replace("-", "")

    try:
        df = fetch_stock_data(stock_codes, start_date, end_date)
        if df.empty:
            return [go.Figure(), go.Figure(), [], []]

        # 特征工程
        features = df.groupby('code').agg({
            'close': ['mean', 'std'],
            'vol': 'mean',
            'pct_chg': ['mean', 'std'],
            'high': lambda x: x.max() / x.min()
        })
        features.columns = ['mean_price', 'price_vol', 'mean_vol', 'pct_mean', 'pct_vol', 'price_range']
        features = features.reset_index()

        # 检查样本数量
        if len(features) < 2:
            return [go.Figure(), go.Figure(), [], []]
        if k > len(features):
            k = max(2, len(features) - 1)

        # 标准化数据
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features.iloc[:, 1:])

        # 动态调整K值
        k = min(k, len(features))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        features['cluster'] = kmeans.fit_predict(scaled)

        # t-SNE可视化 - 动态计算perplexity
        if len(features) > 1:
            perplexity_val = min(30, max(5, len(features) - 1))
            perplexity_val = min(perplexity_val, len(features) - 1)  # 必须小于样本数
            tsne = TSNE(n_components=2, perplexity=perplexity_val, random_state=42)
            tsne_features = tsne.fit_transform(scaled)
            features['tsne_x'] = tsne_features[:, 0]
            features['tsne_y'] = tsne_features[:, 1]
        else:
            features['tsne_x'] = np.nan
            features['tsne_y'] = np.nan

        # 生成聚类图
        if len(features) > 1:
            cluster_fig = px.scatter(
                features, x='tsne_x', y='tsne_y', color='cluster',
                hover_data=['code'], title=f"股票聚类 (K={k}, Perplexity={perplexity_val})"
            )
        else:
            cluster_fig = go.Figure(
                data=[go.Scatter(x=[0], y=[0], mode='markers')],
                layout=dict(title="无法聚类: 样本不足")
            )

        # 生成综合指数图
        index_df = df.groupby('trade_date').apply(
            lambda x: (x['close'] * x['vol']).sum() / x['vol'].sum()
        ).reset_index(name='index')
        index_fig = px.line(index_df, x='trade_date', y='index', title="综合加权指数")

        return [
            cluster_fig,
            index_fig,
            features.to_dict('records'),
            [{'name': col, 'id': col} for col in features.columns]
        ]
    except Exception as e:
        print(f"聚类发生错误: {str(e)}")
        return [go.Figure(), go.Figure(), [], []]

# =========================================
# 决策树分析回调（保持不变）
# =========================================
@app.callback(
    [Output('tree-graph', 'children'),
     Output('feature-importance', 'figure'),
     Output('classification-table', 'data'),
     Output('classification-table', 'columns'),
     Output('investment-advice', 'data')],
    [Input('analyze-btn', 'n_clicks')],
    [State('stock-code', 'value'),
     State('date-range', 'end_date'),
     State('years-slider', 'value'),
     State('threshold', 'value'),
     State('tree-depth', 'value'),
     State('industry-pe', 'value'),
     State('pb-threshold', 'value')]
)
def update_decision_tree(n_clicks, stock_code, end_date, years, threshold,
                         max_depth, industry_pe, pb_threshold):
    if not n_clicks or years is None or max_depth is None:
        return [html.Div(), go.Figure(), [], [], []]

    # 确保输入值有效
    years = int(years)
    if years < 1:
        years = 1
    elif years > 5:
        years = 5
        
    max_depth = int(max_depth)
    if max_depth < 2:
        max_depth = 2
    elif max_depth > 10:
        max_depth = 10

    stock_codes = [c.strip() for c in stock_code.split(',') if c.strip()]
    end_date = pd.to_datetime(end_date)
    start_date = end_date - pd.DateOffset(years=years)

    try:
        # 获取基本面数据
        fundamental_data = []
        for code in stock_codes:
            try:
                df = pro.daily_basic(
                    ts_code=code,
                    start_date=start_date.strftime('%Y%m%d'),
                    end_date=end_date.strftime('%Y%m%d')
                )
                if not df.empty:
                    agg_data = df.agg({
                        'pe': 'mean',
                        'pb': 'mean',
                        'turnover_rate': 'mean'
                    })
                    fundamental_data.append({
                        'code': code,
                        'pe': agg_data['pe'],
                        'pb': agg_data['pb'],
                        'turnover_rate': agg_data['turnover_rate']
                    })
            except:
                continue

        if not fundamental_data:
            return [html.Div("无有效基本面数据", style={'color': 'red'}),
                    go.Figure(), [], [], []]

        features = pd.DataFrame(fundamental_data)

        # 获取价格数据计算标签
        labels = []
        for code in stock_codes:
            try:
                price_df = pro.daily(
                    ts_code=code,
                    start_date=start_date.strftime('%Y%m%d'),
                    end_date=end_date.strftime('%Y%m%d')
                )
                if len(price_df) < 2:
                    continue
                start_price = price_df.iloc[0]['close']
                end_price = price_df.iloc[-1]['close']
                pct_change = (end_price - start_price) / start_price * 100
                labels.append({
                    'code': code,
                    'label': 1 if pct_change >= threshold else 0
                })
            except:
                continue

        if not labels:
            return [html.Div("无有效价格数据", style={'color': 'red'}),
                    go.Figure(), [], [], []]

        label_df = pd.DataFrame(labels)
        full_df = pd.merge(features, label_df, on='code').dropna()

        # 训练决策树
        if len(full_df) < 2:
            return [html.Div("至少需要2只股票进行分析", style={'color': 'red'}),
                    go.Figure(), [], [], []]
                    
        X = full_df[['pe', 'pb', 'turnover_rate']]
        y = full_df['label']
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        clf.fit(X, y)

        # 可视化决策树 - 添加错误处理和资源清理
        try:
            plt.figure(figsize=(20, 10))
            plot_tree(clf, feature_names=X.columns, class_names=['下跌', '上涨'],
                      filled=True, rounded=True, fontsize=12)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            tree_img = html.Img(src='data:image/png;base64,{}'.format(
                base64.b64encode(buf.getvalue()).decode()
            ), style={'width': '100%'})
        except Exception as e:
            tree_img = html.Div(f"决策树可视化错误: {str(e)}", style={'color': 'red'})
        finally:
            plt.close('all')  # 确保释放资源

        # 特征重要性
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
        importance_fig = px.bar(importance, x='importance', y='feature',
                                title='特征重要性排序')

        # 生成建议 - 修复特征名称问题
        full_df['prediction'] = clf.predict(X)
        advice_list = []
        for _, row in full_df.iterrows():
            # 将单行数据转换为DataFrame以保留特征名称
            sample = pd.DataFrame([row[['pe', 'pb', 'turnover_rate']]], 
                                 columns=['pe', 'pb', 'turnover_rate'])
            
            proba = clf.predict_proba(sample)[0]
            confidence_val = max(proba) * 100
            confidence = f"{confidence_val:.1f}%"
            
            advice = "买入" if (row['prediction'] == 1 and row['pe'] < industry_pe) else \
                "卖出" if (row['prediction'] == 0 and row['pb'] > pb_threshold) else "持有"
                
            advice_list.append({
                'code': row['code'],
                'prediction': '上涨' if row['prediction'] == 1 else '下跌',
                'pe': round(row['pe'], 2),
                'industry_pe': industry_pe,
                'pb': round(row['pb'], 2),
                'advice': advice,
                'confidence': confidence
            })

        return [tree_img, importance_fig,
                full_df.to_dict('records'),
                [{'name': col, 'id': col} for col in full_df.columns],
                advice_list]

    except Exception as e:
        print(f"决策树分析错误: {str(e)}")
        return [html.Div(f"分析出现错误: {str(e)}", style={'color': 'red'}),
                go.Figure(), [], [], []]

# =========================================
# 修改后的SVM回调函数
# # =========================================
@app.callback(
    [Output('svm-performance-chart', 'figure'),
     Output('svm-confusion-matrix', 'figure'),  # 新增混淆矩阵输出
     Output('svm-results-table', 'data'),
     Output('svm-results-table', 'columns'),
     Output('svm-accuracy', 'children'),
     Output('svm-advice-table', 'data')],
    [Input('svm-btn', 'n_clicks')],
    [State('stock-code', 'value'),  # 使用主数据展示的股票代码
     State('date-range', 'start_date'),  # 使用主数据展示的开始日期
     State('date-range', 'end_date'),  # 使用主数据展示的结束日期
     State('k-type', 'value'),  # 使用主数据展示的K线类型
     State('train-ratio', 'value'),
     State('kernel-type', 'value')]
)
def run_svm_prediction(n_clicks, stock_code, start_date, end_date, k_type, train_ratio, kernel):
    if n_clicks is None or n_clicks == 0:
        return [go.Figure(), go.Figure(), [], [], "", []]

    stock_codes = [c.strip().upper() for c in stock_code.split(',') if c.strip()]
    start_date_fmt = start_date.replace("-", "")
    end_date_fmt = end_date.replace("-", "")

    # 获取数据
    df_all = fetch_stock_data(stock_codes, start_date_fmt, end_date_fmt, k_type)
    if df_all.empty:
        return [go.Figure(), go.Figure(), [], [], "没有获取到数据", []]

    # 存储每个股票的结果
    all_results = []
    accuracy_list = []
    advice_list = []
    confusion_figs = []  # 存储混淆矩阵图表
    
    for code in stock_codes:
        df = df_all[df_all['code'] == code].copy()
        if df.empty or len(df) < 50:  # 确保有足够的数据
            continue
            
        # 准备特征
        try:
            df_prepared = prepare_features(df)
            if df_prepared.empty:
                continue
        except Exception as e:
            print(f"准备特征失败: {str(e)}")
            continue
            
        # 训练模型
        try:
            train_ratio_decimal = train_ratio / 100.0
            results, accuracy, conf_matrix, class_report, model = train_svm_model(
                df_prepared, kernel, train_ratio_decimal
            )
            
            # 记录准确率
            accuracy_list.append(f"{code}: {accuracy:.2%}")
            
            # 生成混淆矩阵
            conf_fig = go.Figure(data=go.Heatmap(
                z=conf_matrix,
                x=['预测下跌', '预测上涨'],
                y=['实际下跌', '实际上涨'],
                colorscale='Blues',
                text=conf_matrix,
                texttemplate="%{text}",
                textfont={"size":20}
            ))
            conf_fig.update_layout(
                title=f'{code} 混淆矩阵',
                xaxis_title="预测结果",
                yaxis_title="实际结果"
            )
            confusion_figs.append(conf_fig)
            
            # 生成投资建议
            latest_data = df_prepared.iloc[-1]  # 使用最新数据
            prediction, probability, advice = generate_advice(df_prepared, model, latest_data)
            advice_list.append({
                'code': code,
                'prediction': '上涨' if prediction == 1 else '下跌',
                'confidence': f"{probability:.2%}",
                'advice': advice
            })
            
            # 添加股票代码列
            results['stock'] = code
            all_results.append(results)
            
        except Exception as e:
            print(f"股票 {code} SVM预测失败: {str(e)}")
            continue

    # 合并所有结果
    if all_results:
        all_results_df = pd.concat(all_results)
        # 格式化结果表格
        all_results_df['Actual'] = all_results_df['Actual'].map({0: '下跌', 1: '上涨'})
        all_results_df['Predicted'] = all_results_df['Predicted'].map({0: '下跌', 1: '上涨'})
        all_results_df['Probability'] = all_results_df['Probability'].apply(lambda x: f"{x:.2%}")
        all_results_df = all_results_df.rename(columns={
            'Date': '日期', 'Actual': '实际涨跌', 
            'Predicted': '预测涨跌', 'Probability': '置信度', 'stock': '股票代码'
        })
        
        columns = [{'name': col, 'id': col} for col in all_results_df.columns]
        data = all_results_df.to_dict('records')
    else:
        data = []
        columns = []
    
    # 创建性能图表
    fig = go.Figure()
    for code in stock_codes:
        stock_results = all_results_df[all_results_df['股票代码'] == code]
        if not stock_results.empty:
            # 为每个股票添加实际值线
            fig.add_trace(go.Scatter(
                x=stock_results['日期'], 
                y=stock_results['实际涨跌'],
                name=f"{code}实际",
                mode='lines',
                line=dict(color='blue', width=2)
            ))
            # 为每个股票添加预测值点
            fig.add_trace(go.Scatter(
                x=stock_results['日期'], 
                y=stock_results['预测涨跌'],
                name=f"{code}预测",
                mode='markers',
                marker=dict(size=8, color='red')
            ))
    
    fig.update_layout(
        title="SVM预测结果",
        xaxis_title="日期",
        yaxis_title="涨跌",
        height=400
    )
    
    # 准确率文本
    accuracy_text = "准确率: " + "; ".join(accuracy_list) if accuracy_list else "无准确率数据"
    
    # 生成性能图表（混淆矩阵）
    conf_fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=['预测下跌', '预测上涨'],
        y=['实际下跌', '实际上涨'],
        colorscale='Blues',
        text=conf_matrix,
        texttemplate="%{text}",
        textfont={"size":20}
    ))
    conf_fig.update_layout(
        title=f'混淆矩阵 (准确率: {accuracy:.2%})',
        xaxis_title="预测结果",
        yaxis_title="实际结果"
    )

    return [
        fig,
        conf_fig,
        data,
        columns,
        accuracy_text,
        advice_list
    ]

# =========================================
# 5. 云图回调函数
# =========================================
@app.callback(
    Output('cloud-plot', 'figure'),
    [Input('cloud-btn', 'n_clicks')],
    [State('board-selector', 'value'),
     State('stock-count', 'value')]
)
def update_cloud(n_clicks, boards, stock_count):
    if not n_clicks or not boards:
        return go.Figure()
    
    all_stocks = []
    for board in boards:
        board_stocks = get_board_stocks(board, stock_count)
        all_stocks.extend(board_stocks)
    
    # 获取股票最新数据
    end_date = pd.to_datetime('today').strftime('%Y%m%d')
    start_date = (pd.to_datetime('today') - timedelta(days=30)).strftime('%Y%m%d')
    
    try:
        # 获取股票数据
        df = fetch_stock_data(all_stocks, start_date, end_date)
        if df.empty:
            return go.Figure()
        
        # 获取最新数据
        latest_df = df.sort_values('trade_date').groupby('code').last().reset_index()
        
        # 添加板块信息
        board_map = {
            'sh': '上证', 'sz': '深证', 'cyb': '创业板', 'kcb': '科创板', 'bj': '北交所'
        }
        latest_df['board'] = latest_df['code'].apply(
            lambda x: board_map.get(x.split('.')[1].lower(), '其他')
        )
        
        # 创建云图
        fig = px.scatter(
            latest_df,
            x='code',
            y='pct_chg',
            color='pct_chg',
            color_continuous_scale='RdYlGn',
            size='vol',
            hover_data=['code', 'close', 'pct_chg', 'board'],
            title='股票云图',
            labels={'code': '股票代码', 'pct_chg': '涨跌幅(%)', 'board': '板块'}
        )
        
        # 更新布局
        fig.update_layout(
            xaxis_title='股票代码',
            yaxis_title='涨跌幅(%)',
            height=600,
            hovermode='closest',
            coloraxis_colorbar=dict(title="涨跌幅(%)")
        )
        
        return fig
    
    except Exception as e:
        print(f"云图生成错误: {str(e)}")
        return go.Figure()

# =========================================
# 新闻分析回调函数
# =========================================
@app.callback(
    [Output('sentiment-trend-chart', 'figure'),
     Output('event-calendar', 'data'),
     Output('news-summary', 'children')],
    [Input('analyze-news-btn', 'n_clicks')],
    [State('news-keyword', 'value'),
     State('news-date-range', 'start_date'),
     State('news-date-range', 'end_date'),
     State('sentiment-filter', 'value'),
     State('impact-slider', 'value')]
)
def update_news_analysis(n_clicks, keyword, start_date, end_date, sentiment_filter, impact_threshold):
    if not n_clicks or not keyword:
        return go.Figure(), [], "请输入关键词"
    
    # 1. 获取新闻数据（实际应用替换为真实API）
    news_data = get_news_data(keyword, start_date, end_date)
    
    # 2. 百炼API分析情感
    results = []
    events = []
    news_cards = []
    
    for news in news_data:
        # 调用情感分析API
        analysis = analyze_news_sentiment(news['content'])
        
        if 'error' in analysis:
            # 错误处理 - 使用随机数据作为后备
            sentiment = random.choice(['positive', 'neutral', 'negative'])
            confidence = round(random.uniform(0.55, 0.95), 2)
            impact = random.randint(3, 10)
        else:
            sentiment = analysis['sentiment']
            confidence = analysis['confidence']
            impact = analysis['impact_score']
        
        # 应用过滤条件
        if sentiment_filter != 'all' and sentiment != sentiment_filter:
            continue
        if impact < impact_threshold:
            continue
        
        # 记录分析结果
        results.append({
            'date': news['date'],
            'sentiment': sentiment,
            'confidence': confidence,
            'impact': impact
        })
        
        # 收集重大事件
        if impact > 7:
            events.append({
                'date': news['date'],
                'event_type': news['title'],
                'impact': impact
            })
        
        # 生成新闻卡片
        color_map = {'positive': 'success', 'neutral': 'warning', 'negative': 'danger'}
        badge_color = color_map.get(sentiment, 'secondary')
        
        news_card = dbc.Card(
            [
                dbc.CardHeader([
                    html.Span(news['date'], className="mr-2"),
                    dbc.Badge(sentiment, color=badge_color, className="mr-1"),
                    dbc.Badge(f"影响力: {impact}", color="info")
                ]),
                dbc.CardBody([
                    html.H5(news['title'], className="card-title"),
                    html.P(news['content'], className="card-text"),
                    html.Footer([
                        html.Small(news['source'], className="text-muted"),
                        html.Span(f" | 置信度: {confidence:.2f}", className="ml-2")
                    ])
                ])
            ],
            className="mb-3",
            style={'borderLeft': f'5px solid'},  # 颜色根据情感动态设置
            color=badge_color,
            outline=True
        )
        news_cards.append(news_card)
    
    # 3. 构建情感趋势图
    if results:
        df = pd.DataFrame(results)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # 按日期分组计算平均情感值
        df_grouped = df.groupby('date')['impact'].mean().reset_index()
        
        fig = px.line(
            df_grouped, 
            x='date', 
            y='impact',
            title=f"{keyword} 新闻情感趋势",
            labels={'impact': '影响力指数', 'date': '日期'}
        )
        fig.update_layout(
            xaxis_title='日期',
            yaxis_title='影响力指数',
            hovermode='x unified',
            template='plotly_white'
        )
        
        # 添加散点图显示每日新闻数量
        count_df = df.groupby('date').size().reset_index(name='count')
        fig.add_trace(go.Scatter(
            x=count_df['date'], 
            y=count_df['count'] * 0.8 + 1,  # 调整到合适位置
            mode='markers',
            marker=dict(size=count_df['count'] * 3, color='rgba(255, 182, 193, 0.5)'),
            name='新闻数量',
            yaxis='y2'
        ))
        
        fig.update_layout(
            yaxis2=dict(
                title='新闻数量',
                overlaying='y',
                side='right',
                range=[0, max(count_df['count']) * 1.2]
            )
        )
    else:
        fig = go.Figure()
        fig.update_layout(
            title='无符合条件的数据',
            xaxis_title='日期',
            yaxis_title='影响力指数',
            template='plotly_white'
        )
    
    # 4. 准备事件日历数据
    event_data = sorted(events, key=lambda x: x['date'], reverse=True)
    
    return fig, event_data, news_cards

# =========================================
# LSTM预测回调函数
# =========================================
@app.callback(
    [Output('lstm-forecast-chart', 'figure'),
     Output('lstm-results-table', 'columns'),
     Output('lstm-results-table', 'data'),
     Output('lstm-advice-table', 'data')],
    [Input('lstm-btn', 'n_clicks')],
    [State('stock-code', 'value'),
     State('date-range', 'start_date'),
     State('date-range', 'end_date'),
     State('k-type', 'value'),
     State('forecast-days', 'value'),
     State('window-size', 'value'),
     State('lstm-layers', 'value'),
     State('epochs', 'value')]
)
def update_lstm_predictions(n_clicks, stock_codes, start_date, end_date, k_type, 
                           forecast_days, window_size, lstm_layers, epochs):
    if n_clicks is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    # 获取股票数据
    codes = [code.strip() for code in stock_codes.split(',')]
    df = fetch_stock_data(codes, start_date, end_date, k_type)
    
    if df.empty:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    # 对每只股票进行预测
    all_results = []
    advice_data = []
    
    fig = go.Figure()
    
    for code in codes:
        stock_df = df[df['code'] == code].sort_values('trade_date')
        if len(stock_df) < window_size + 10:  # 确保有足够的数据
            continue
        
        # 执行LSTM预测
        predictions = lstm_predict(
            stock_df, 
            forecast_days=forecast_days,
            window_size=window_size,
            lstm_layers=lstm_layers,
            epochs=epochs
        )
        
        if not predictions:
            continue
        
        # 添加历史数据到图表
        fig.add_trace(go.Scatter(
            x=stock_df['trade_date'],
            y=stock_df['close'],
            mode='lines',
            name=f'{code} 历史价格',
            line=dict(width=2)
        ))
        
        # 添加预测数据到图表
        pred_dates = [pd.to_datetime(p['date']) for p in predictions]
        pred_prices = [p['predicted_price'] for p in predictions]
        
        fig.add_trace(go.Scatter(
            x=pred_dates,
            y=pred_prices,
            mode='lines+markers',
            name=f'{code} 预测价格',
            line=dict(color='red', width=2, dash='dot')
        ))
        
        # 添加置信区间（灰色区域）
        min_price = min(pred_prices) * 0.98
        max_price = max(pred_prices) * 1.02
        fig.add_trace(go.Scatter(
            x=pred_dates + pred_dates[::-1],
            y=[min_price] * len(pred_dates) + [max_price] * len(pred_dates),
            fill='toself',
            fillcolor='rgba(200,200,200,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # 准备表格数据
        for p in predictions:
            all_results.append({
                'code': code,
                'date': p['date'],
                'predicted_price': round(p['predicted_price'], 2),
                'confidence': f"{p['confidence']*100:.1f}%"
            })
        
        # 生成投资建议
        last_price = stock_df['close'].iloc[-1]
        predicted_change = (pred_prices[-1] - last_price) / last_price * 100
        
        if predicted_change > 15:
            advice = "强烈买入"
            confidence = "高"
        elif predicted_change > 5:
            advice = "买入"
            confidence = "中高"
        elif predicted_change > -2:
            advice = "持有"
            confidence = "中"
        elif predicted_change > -10:
            advice = "卖出"
            confidence = "中低"
        else:
            advice = "强烈卖出"
            confidence = "低"
            
        advice_data.append({
            'code': code,
            'predicted_change': f"{predicted_change:.2f}%",
            'confidence': confidence,
            'advice': advice
        })
    
    # 设置图表布局
    fig.update_layout(
        title='股票价格预测',
        xaxis_title='日期',
        yaxis_title='价格',
        legend_title='股票代码',
        hovermode='x unified'
    )
    
    # 设置表格列
    columns = [
        {'name': '代码', 'id': 'code'},
        {'name': '日期', 'id': 'date'},
        {'name': '预测价格', 'id': 'predicted_price'},
        {'name': '置信度', 'id': 'confidence'}
    ]
    
    return fig, columns, all_results, advice_data

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)