# 股票行情分析预测系统 v1.4

## 项目概述

本系统是一个基于Dash框架的股票分析预测平台，集成了多种机器学习模型和技术分析工具，提供全面的股票市场数据可视化和预测功能。

## 功能特性

### 核心功能模块

- **数据可视化模块**
  - 实时股票K线图展示
  - 多种技术指标分析（MACD、RSI、KDJ、布林带等）
  - 成交量与价格趋势联动分析

- **机器学习分析模块**
  - K-Means聚类分析
  - 决策树价值投资分析
  - SVM股票涨跌预测
  - LSTM时序价格预测

- **辅助分析工具**
  - 股票云图可视化
  - 新闻情感分析
  - 综合加权指数计算

- **系统特性**
  - 响应式用户界面
  - 模块化设计
  - 多标签页导航

## 技术栈

- **前端框架**: Dash + Dash Bootstrap Components
- **可视化库**: Plotly + Matplotlib
- **数据处理**: Pandas + Numpy
- **机器学习**: Scikit-learn + TensorFlow
- **数据源**: Tushare Pro API
- **NLP分析**: Dashscope API

## 安装指南

1. 克隆仓库
```bash
git clone https://github.com/zhankenan/stock_analysis_app.git
cd stock_analysis_app
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置API密钥
在`utils.py`中设置您的Tushare Pro和API密钥

## 使用说明

1. 启动应用
```bash
python app.py
```

2. 访问应用
打开浏览器访问 `http://localhost:8051`

3. 功能导航
- 主界面: 股票数据查询和技术指标分析
- 聚类分析: 股票聚类和特征分析
- 决策树: 价值投资决策分析
- SVM预测: 基于支持向量机的涨跌预测
- LSTM预测: 基于长短期记忆网络的价格预测

## 贡献指南

欢迎提交Pull Request或Issue报告问题。请确保代码风格一致并通过测试。

## 许可证

MIT License
