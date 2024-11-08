import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from ta.momentum import RSIIndicator  # Teknik göstergeler için 'ta' kütüphanesi
from ta.volatility import BollingerBands
from datetime import datetime, timedelta

def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Hareketli ortalamalar (MA)
    stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
    
    # RSI (Relative Strength Index)
    rsi = RSIIndicator(stock_data['Close'])
    stock_data['RSI'] = rsi.rsi()
    
    # Bollinger Band
    bb = BollingerBands(stock_data['Close'], window=20, window_dev=2)
    stock_data['BB_upper'] = bb.bollinger_hband()
    stock_data['BB_lower'] = bb.bollinger_lband()

    # Nan değerleri dolduralım
    stock_data.fillna(0, inplace=True)
    
    return stock_data

def train_model(data, model_type='linear'):
    # Özellikler ve hedef değişken
    X = data[['Open', 'High', 'Low', 'Volume', 'MA_20', 'MA_50', 'RSI', 'BB_upper', 'BB_lower']]
    y = data['Close']
    
    # Model oluşturma ve eğitim
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'xgboost':
        model = XGBRegressor(n_estimators=100, random_state=42)
    
    model.fit(X, y)
    return model

def predict_future_prices(model, past_data, future_dates):
    # Geçmiş veriden son günün özelliklerini kullan
    last_row = past_data.iloc[-1]
    last_features = np.array([last_row['Open'], last_row['High'], last_row['Low'], last_row['Volume'], 
                              last_row['MA_20'], last_row['MA_50'], last_row['RSI'], 
                              last_row['BB_upper'], last_row['BB_lower']]).reshape(1, -1)
    
    future_prices = []
    for _ in future_dates:
        # Tahmin yap
        predicted_close = model.predict(last_features)[0]
        future_prices.append(predicted_close)
        
        # Gelecek gün için özellikleri güncelle
        last_features = np.array([predicted_close, predicted_close, predicted_close, last_row['Volume'],
                                  predicted_close, predicted_close, last_row['RSI'],
                                  last_row['BB_upper'], last_row['BB_lower']]).reshape(1, -1)
    
    return future_prices

def backtest_model(model, data, train_size_ratio=0.8):
    # Veriyi eğitim ve test olarak ayıralım
    train_size = int(len(data) * train_size_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Eğitim verisi ile model eğitimi
    X_train = train_data[['Open', 'High', 'Low', 'Volume', 'MA_20', 'MA_50', 'RSI', 'BB_upper', 'BB_lower']]
    y_train = train_data['Close']
    
    X_test = test_data[['Open', 'High', 'Low', 'Volume', 'MA_20', 'MA_50', 'RSI', 'BB_upper', 'BB_lower']]
    y_test = test_data['Close']
    
    model.fit(X_train, y_train)
    
    # Test verisinde tahmin yapalım
    y_pred = model.predict(X_test)
    
    # Hata hesapları
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    return y_test, y_pred, mae, rmse

# Mevcut tarih
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')  # Son 3 ayın verisi

# Gerçek borsa verileri
ticker = 'TUPRS.IS' 
data = get_stock_data(ticker, start_date, end_date)

# Model eğitimi ve backtesting (Random Forest)
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
y_test_rf, y_pred_rf, mae_rf, rmse_rf = backtest_model(model_rf, data)
print(f"Random Forest MAE: {mae_rf}, RMSE: {rmse_rf}")

# Model eğitimi ve backtesting (XGBoost)
model_xgb = XGBRegressor(n_estimators=100, random_state=42)
y_test_xgb, y_pred_xgb, mae_xgb, rmse_xgb = backtest_model(model_xgb, data)
print(f"XGBoost MAE: {mae_xgb}, RMSE: {rmse_xgb}")

# Model eğitimi ve backtesting (Linear Regression)
model_lr = LinearRegression()
y_test_lr, y_pred_lr, mae_lr, rmse_lr = backtest_model(model_lr, data)
print(f"Linear Regression MAE: {mae_lr}, RMSE: {rmse_lr}")

# Backtesting sonuçlarını grafik üzerinde gösterme
plt.figure(figsize=(14, 7))
plt.plot(data.index[-len(y_test_rf):], y_test_rf, label='Gerçek Kapanış Fiyatı', marker='o', color='blue')
plt.plot(data.index[-len(y_pred_rf):], y_pred_rf, label='Random Forest Tahmini', marker='o', linestyle='--', color='green')
plt.plot(data.index[-len(y_pred_xgb):], y_pred_xgb, label='XGBoost Tahmini', marker='o', linestyle='--', color='red')
plt.plot(data.index[-len(y_pred_lr):], y_pred_lr, label='Linear Regression Tahmini', marker='o', linestyle='--', color='purple')
plt.title('Backtesting: Gerçek ve Tahmin Edilen Kapanış Fiyatları')
plt.xlabel('Tarih')
plt.ylabel('Fiyat')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
