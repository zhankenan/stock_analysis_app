import dash
import dash_bootstrap_components as dbc
from dash import html
from components import create_layout

# 初始化应用
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "股票行情分析预测系统"

# 设置布局
app.layout = create_layout()

# 在main模块中运行
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)