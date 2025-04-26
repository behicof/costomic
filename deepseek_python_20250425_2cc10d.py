
# -*- coding: utf-8 -*-
"""
سیستم ترید خودکار متصل به MetaTrader 5
نسخه: 2.0
امکانات:
- اتصال زنده به MetaTrader 5
- نمایش لحظه‌ای قیمت در داشبورد
- سیگنال‌دهی خودکار خرید/فروش
- اجرای خودکار معاملات
- محاسبه سود/زیان بلادرنگ
"""

import time
import logging
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
import MetaTrader5 as mt5
from sklearn.preprocessing import MinMaxScaler
import joblib

# 1. تنظیمات اولیه
logging.basicConfig(level=logging.INFO)
SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
MODEL_PATH = "xgb_model.pkl"
SCALER_PATH = "scaler.pkl"

# 2. کلاس اتصال به MetaTrader
class MT5Connector:
    def __init__(self, account, password, server):
        self.account = account
        self.password = password
        self.server = server
        self.connected = False
        
    def connect(self):
        """اتصال به حساب MetaTrader"""
        if not mt5.initialize():
            logging.error("خطا در اتصال به MetaTrader 5")
            return False
            
        authorized = mt5.login(
            login=self.account,
            password=self.password,
            server=self.server
        )
        
        if authorized:
            self.connected = True
            logging.info("اتصال موفقیت‌آمیز به حساب معاملاتی")
            return True
        return False

    def get_real_time_data(self):
        """دریافت داده‌های لحظه‌ای"""
        if not self.connected:
            return None
            
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 100)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

# 3. کلاس سیستم ترید خودکار
class AutoTrader:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.open_positions = pd.DataFrame(columns=['time', 'type', 'price', 'volume'])
        self.trade_history = pd.DataFrame(columns=['open_time', 'close_time', 'type', 'profit'])
        
    def generate_features(self, df):
        """تولید ویژگی‌های مدل"""
        df['Open_Diff'] = df['open'].diff()
        df['Range'] = df['high'] - df['low']
        df['RSI'] = self.calculate_rsi(df['close'])
        return df[['Open_Diff', 'Range', 'RSI']].iloc[-1].values.reshape(1, -1)
        
    def calculate_rsi(self, series, period=14):
        """محاسبه RSI"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
        
    def predict_signal(self, features):
        """پیش‌بینی سیگنال"""
        scaled_features = self.scaler.transform(features)
        prediction = self.model.predict(scaled_features)
        return "BUY" if prediction[0] == 1 else "SELL"
        
    def execute_trade(self, signal, current_price):
        """اجرای معامله"""
        trade_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": 0.1,
            "type": trade_type,
            "price": current_price,
            "deviation": 20,
            "magic": 234000,
            "comment": "AutoTrade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"خطا در اجرای معامله: {result.comment}")
            return False
            
        self.record_trade(result, signal)
        return True
        
    def record_trade(self, result, signal):
        """ثبت معامله"""
        new_trade = {
            'time': pd.Timestamp.now(),
            'type': signal,
            'price': result.price,
            'volume': result.volume
        }
        self.open_positions = self.open_positions.append(new_trade, ignore_index=True)
        
# 4. راه‌اندازی داشبورد
app = Dash(__name__)

app.layout = html.Div([
    html.H1("سیستم ترید خودکار طلا", style={'textAlign': 'center'}),
    
    dcc.Interval(
        id='interval-component',
        interval=10*1000,  # بروزرسانی هر 10 ثانیه
        n_intervals=0
    ),
    
    html.Div([
        dcc.Graph(id='price-chart', style={'width': '60%', 'display': 'inline-block'}),
        html.Div([
            html.H3("سیگنال فعلی:", id='current-signal'),
            html.H3("موقعیت‌های باز:", id='open-positions'),
            html.H3("سود/زیان کل:", id='total-pnl'),
        ], style={'width': '35%', 'float': 'right'})
    ]),
    
    html.Div(id='hidden-div', style={'display': 'none'})
])

# 5. Callbackهای تعاملی
@app.callback(
    [Output('price-chart', 'figure'),
     Output('current-signal', 'children'),
     Output('open-positions', 'children'),
     Output('total-pnl', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    # دریافت داده‌های جدید
    df = connector.get_real_time_data()
    current_price = mt5.symbol_info_tick(SYMBOL).ask
    
    # تولید سیگنال
    features = trader.generate_features(df)
    signal = trader.predict_signal(features)
    
    # اجرای معامله اگر شرایط فراهم باشد
    if should_execute_trade(signal):
        trader.execute_trade(signal, current_price)
    
    # به‌روزرسانی نمودارها
    fig = go.Figure(data=[
        go.Candlestick(
            x=df['time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close']
        )
    ])
    
    # محاسبه سود/زیان
    pnl = calculate_pnl(current_price)
    
    return (
        fig,
        f"سیگنال فعلی: {signal}",
        f"موقعیت‌های باز: {len(trader.open_positions)}",
        f"سود/زیان کل: {pnl}$"
    )

def should_execute_trade(signal):
    """شرایط اجرای معامله"""
    # افزودن منطق مدیریت ریسک
    return True  # نمونه ساده

def calculate_pnl(current_price):
    """محاسبه سود/زیان"""
    if trader.open_positions.empty:
        return 0.0
    return sum(
        (current_price - row['price']) * row['volume'] * 
        (1 if row['type'] == 'BUY' else -1)
        for _, row in trader.open_positions.iterrows()
    )

# 6. راه‌اندازی سیستم
if __name__ == "__main__":
    # اتصال به MetaTrader
    connector = MT5Connector(
        account=91956651,
        password="X*N8FsHz",
        server="MetaQuotes-Demo"
    )
    
    if not connector.connect():
        raise ConnectionError("اتصال به MetaTrader ناموفق بود")
        
    # راه‌اندازی تریدر
    trader = AutoTrader()
    
    # اجرای داشبورد
    app.run_server(debug=False, port=8050)