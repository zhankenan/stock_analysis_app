import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from datetime import datetime
from datetime import timedelta
import pandas as pd
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

#svm模块
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

def create_layout():
    # 完整布局结构（同原代码）
    return html.Div([
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
        
        # 聚类分析模块 - 初始隐藏
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
        
        # 决策树模块 - 初始隐藏
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
        # 新增云图模块（简化版）
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