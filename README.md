# 股票行情分析预测系统

## 项目概述

股票行情分析预测系统是一个基于Dash框架构建的专业金融分析平台，集成了多种机器学习算法和技术指标分析工具，为投资者提供全面的股票市场分析、预测和可视化功能。系统支持多维度数据分析，包括技术指标可视化、聚类分析、决策树估值、SVM预测、LSTM时序预测等核心功能，并结合新闻情感分析提供综合投资建议。

## 系统架构

graph TD
    A[Dash前端] --> B[核心分析模块]
    A --> C[预测模型模块]
    A --> D[可视化模块]
    B --> E[技术指标分析]
    B --> F[聚类分析]
    B --> G[决策树估值]
    C --> H[SVM分类预测]
    C --> I[LSTM时序预测]
    D --> J[K线图表]
    D --> K[股票云图]
    D --> L[新闻情感分析]
功能模块
1. 主行情分析
多股票并行分析
自定义K线类型（日K/周K/月K）
技术指标叠加（MACD/KDJ/RSI/布林带/均线）
实时数据表格展示
2. 聚类分析
K-means股票聚类
t-SNE降维可视化
板块综合指数计算
聚类结果表格展示
3. 决策树估值
基本面指标分析（PE/PB/换手率）
动态决策树可视化
特征重要性分析
价值投资建议生成
4. SVM预测
支持向量机分类模型
多核函数选择（线性/多项式/径向基/Sigmoid）
混淆矩阵可视化
涨跌预测与置信度评估
5. LSTM预测
长短时记忆网络预测
可配置回看窗口
多层LSTM架构
价格预测区间可视化
6. 股票云图
多板块股票筛选（主板/创业板/科创板/北交所）
成交量加权可视化
涨跌幅颜色编码
交互式股票探索
7. 新闻事件分析
金融新闻情感分析（百炼API）
情感趋势可视化
重大事件日历
影响力评分系统
技术栈
核心框架
​​Dash​​ - 主应用框架
​​Dash Bootstrap Components​​ - UI组件库
​​Plotly​​ - 交互式可视化
数据处理
​​Pandas​​ - 数据分析
​​NumPy​​ - 数值计算
​​Tushare​​ - 金融数据接口
机器学习
​​Scikit-learn​​:
KMeans聚类
SVM分类
决策树
特征工程
​​TensorFlow/Keras​​ - LSTM模型
辅助工具
​​Seaborn/Matplotlib​​ - 静态可视化
​​TSNE​​ - 降维算法
​​Dashscope​​ - 新闻情感分析API
安装指南
前置要求
Python 3.8+
Tushare Pro账号（获取API token）
阿里云百炼API Key（新闻分析功能）
安装步骤
克隆仓库：
git clone https://github.com/your-repo/stock-analysis-system.git
cd stock-analysis-system
安装依赖：
pip install -r requirements.txt
配置环境变量：
创建 .env 文件并添加：
TU_SHARE_TOKEN=your_tushare_token
DASHSCOPE_API_KEY=your_dashscope_key
运行应用：
python app.py
访问应用：
http://localhost:8051
文件结构
stock-analysis-system/
├── app.py                 # 主应用入口
├── callbacks.py           # 回调函数处理
├── utils.py               # 工具函数库
├── config.py              # 配置常量
├── requirements.txt       # 依赖列表
├── .env.example           # 环境变量示例
├── assets/                # 静态资源
│   ├── custom.css         # 自定义样式
├── data_processing/       # 数据处理模块
│   ├── data_fetcher.py    # 数据获取
│   ├── feature_engineer.py # 特征工程
├── models/                # 机器学习模型
│   ├── clustering.py      # 聚类模型
│   ├── svm_model.py       # SVM分类
│   ├── lstm_model.py      # LSTM预测
│   └── decision_tree.py   # 决策树模型
└── visualization/         # 可视化模块
    ├── chart_builder.py   # 图表生成
    └── cloud_plot.py      # 股票云图
使用示例
1. 多股票技术分析
# 输入股票代码
stock_codes = '600519.SH,000001.SZ,300750.SZ'

# 设置分析参数
start_date = '2023-01-01'
end_date = '2024-01-01'
indicators = ['macd', 'rsi', 'boll']
2. LSTM价格预测
# 配置LSTM参数
lstm_params = {
    'forecast_days': 7,
    'window_size': 30,
    'lstm_layers': 2,
    'epochs': 50
}

# 获取预测结果
predictions = lstm_predict(stock_data, **lstm_params)
3. 新闻情感分析
# 分析新闻情感
news_analysis = analyze_news_sentiment("宁德时代发布新一代麒麟电池，能量密度提升15%")

# 返回结果
{
    'sentiment': 'positive',
    'confidence': 0.87,
    'impact_score': 8,
    'keywords': '宁德时代,麒麟电池,能量密度'
}
API文档
数据获取函数
fetch_stock_data(stock_codes, start_date, end_date, k_type='D')

功能：获取多股票历史数据
参数：
stock_codes: 股票代码列表（逗号分隔）
start_date: 开始日期（YYYY-MM-DD）
end_date: 结束日期（YYYY-MM-DD）
k_type: K线类型（D:日线, W:周线, M:月线）
返回：包含历史数据的DataFrame
技术指标计算
calculate_macd(df)

功能：计算MACD指标
参数：包含收盘价的DataFrame
返回：(MACD线, 信号线, 柱状图)
calculate_kdj(df, n=9, m1=3, m2=3)

功能：计算KDJ指标
返回：(K值, D值, J值)
性能优化
​​数据缓存机制​​：
使用内存缓存频繁访问的股票数据
减少Tushare API调用次数
​​并行计算​​：
多股票分析使用Joblib并行处理
LSTM预测启用GPU加速
​​增量更新​​：
每日只获取增量数据
模型增量训练
​​资源管理​​：
# 限制线程资源
os.environ['OMP_NUM_THREADS'] = '1'
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
贡献指南
欢迎通过Issue或Pull Request贡献代码：

Fork项目仓库
创建特性分支（git checkout -b feature/AmazingFeature）
提交更改（git commit -m 'Add some AmazingFeature'）
推送到分支（git push origin feature/AmazingFeature）
创建Pull Request
许可证
本项目采用 MIT License

致谢
Tushare提供金融数据支持
阿里云百炼提供自然语言处理API
Dash社区提供前端框架支持
