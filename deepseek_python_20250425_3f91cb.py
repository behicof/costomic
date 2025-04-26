# train_model.py
import pandas as pd
import talib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib

# بارگذاری داده‌های تاریخی
data = pd.read_csv('DAT_MS_XAUUSD_M1_202404.csv',
                   names=['Symbol', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
data['Time'] = pd.to_datetime(data['Time'], format='%Y%m%d%H%M')
data.set_index('Time', inplace=True)

# مهندسی ویژگی‌ها
data['Open_Diff'] = data['Open'].diff()
data['Range'] = data['High'] - data['Low']
data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
data['Target'] = (data['Close'] > data['Close'].shift(-1)).astype(int)
data.dropna(inplace=True)

# تقسیم داده
X = data[['Open_Diff', 'Range', 'RSI']]
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# آموزش مدل
model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
model.fit(X_train, y_train)

# ذخیره مدل
joblib.dump(model, 'xgb_model.pkl')
print("مدل با موفقیت ذخیره شد!")