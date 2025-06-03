# utils.py
import tushare as ts
import pandas as pd
import numpy as np
import os
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import random
import dashscope
from datetime import datetime, timedelta
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
import plotly.graph_objs as go
import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context


# 初始化Tushare Pro
TOKEN = ''
ts.set_token(TOKEN)
pro = ts.pro_api()

# 配置百炼API Key
os.environ['DASHSCOPE_API_KEY'] = ''

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