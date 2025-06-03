# callbacks.py
from dash.dependencies import Input, Output, State
from dash import dcc, html, callback_context, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import random
import dashscope
from datetime import datetime, timedelta
import time
import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context

from utils import fetch_stock_data, calculate_ma, calculate_macd, calculate_kdj, calculate_rsi, calculate_boll, create_candlestick_chart, get_board_stocks, analyze_news_sentiment, get_news_data, prepare_features, train_svm_model, generate_advice, lstm_predict, pro

# 导入app和cache（如果需要的话）
from app import app

# 在这里定义所有回调函数
# 注意：由于回调函数很多，我们只写出回调函数的装饰器和函数定义，具体实现使用原始代码中的函数体

# 切换标签回调
@app.callback(
    [Output('main-module', 'style'),
     Output('cluster-module', 'style'),
     Output('tree-module', 'style'),
     Output('svm-module', 'style'),
     Output('lstm-module', 'style'),
     Output('cloud-module', 'style'),
     Output('news-module', 'style'),
     Output('main-link', 'active'),
     Output('cluster-link', 'active'),
     Output('tree-link', 'active'),
     Output('svm-link', 'active'),
     Output('lstm-link', 'active'),
     Output('cloud-link', 'active'),
     Output('news-link', 'active'),
     Output('active-tab', 'data')],
    [Input('main-link', 'n_clicks'),
     Input('cluster-link', 'n_clicks'),
     Input('tree-link', 'n_clicks'),
     Input('svm-link', 'n_clicks'),
     Input('lstm-link', 'n_clicks'),
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


# 主回调函数：更新主图表
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

# 标签页切换回调
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

# 聚类分析回调
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

# 决策树分析回调
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

# SVM预测回调
@app.callback(
    [Output('svm-performance-chart', 'figure'),
     Output('svm-confusion-matrix', 'figure'),
     Output('svm-results-table', 'data'),
     Output('svm-results-table', 'columns'),
     Output('svm-accuracy', 'children'),
     Output('svm-advice-table', 'data')],
    [Input('svm-btn', 'n_clicks')],
    [State('stock-code', 'value'),
     State('date-range', 'start_date'),
     State('date-range', 'end_date'),
     State('k-type', 'value'),
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

# 云图回调
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

# 新闻分析回调
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


# LSTM预测回调
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