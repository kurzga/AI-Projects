import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Verileri almak için kullanılan fonksiyon
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Teknik göstergeler
    stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
    rsi = RSIIndicator(stock_data['Close'])
    stock_data['RSI'] = rsi.rsi()
    bb = BollingerBands(stock_data['Close'], window=20, window_dev=2)
    stock_data['BB_upper'] = bb.bollinger_hband()
    stock_data['BB_lower'] = bb.bollinger_lband()
    
    stock_data.fillna(0, inplace=True)
    return stock_data

# Modeli eğitmek için kullanılan fonksiyon
def train_model(data, model_type='linear'):
    X = data[['Open', 'High', 'Low', 'Volume', 'MA_20', 'MA_50', 'RSI', 'BB_upper', 'BB_lower']]
    y = data['Close']
    
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'xgboost':
        model = XGBRegressor(n_estimators=100, random_state=42)
    
    model.fit(X, y)
    return model

# Gelecek fiyatları tahmin etmek için kullanılan fonksiyon
def predict_future_prices(model, past_data, future_dates):
    last_row = past_data.iloc[-1]
    last_features = np.array([last_row['Open'], last_row['High'], last_row['Low'], last_row['Volume'], 
                              last_row['MA_20'], last_row['MA_50'], last_row['RSI'], 
                              last_row['BB_upper'], last_row['BB_lower']]).reshape(1, -1)
    
    future_prices = []
    for _ in future_dates:
        predicted_close = model.predict(last_features)[0]
        future_prices.append(predicted_close)
        last_features = np.array([predicted_close, predicted_close, predicted_close, last_row['Volume'],
                                  predicted_close, predicted_close, last_row['RSI'],
                                  last_row['BB_upper'], last_row['BB_lower']]).reshape(1, -1)
    
    return future_prices

# Backtesting fonksiyonu: geçmiş verilerle tahmin ve karşılaştırma
def backtest_model(model, data, test_size=0.2):
    split_index = int(len(data) * (1 - test_size))
    train_data = data[:split_index]
    test_data = data[split_index:]
    
    model.fit(train_data[['Open', 'High', 'Low', 'Volume', 'MA_20', 'MA_50', 'RSI', 'BB_upper', 'BB_lower']], train_data['Close'])
    
    test_X = test_data[['Open', 'High', 'Low', 'Volume', 'MA_20', 'MA_50', 'RSI', 'BB_upper', 'BB_lower']]
    test_y = test_data['Close']
    
    predictions = model.predict(test_X)
    
    mae = mean_absolute_error(test_y, predictions)
    r2 = r2_score(test_y, predictions)
    
    # Gerçek fiyatlar ve tahminleri karşılaştıralım
    results = pd.DataFrame({
        'Date': test_data.index,
        'Real_Close': test_y,
        'Predicted_Close': predictions
    }).set_index('Date')
    
    return mae, r2, results

# Tarih aralıkları
today = datetime.now()
start_date = (today - timedelta(days=365)).strftime('%Y-%m-%d')  # 1 yıllık veri
end_date = today.strftime('%Y-%m-%d')

# Gerçek borsa verileri
ticker = 'THYAO.IS'
data = get_stock_data(ticker, start_date, end_date)

# Model eğitimi ve backtesting (RandomForest)
model_rf = train_model(data, model_type='random_forest')
mae_rf, r2_rf, results_rf = backtest_model(model_rf, data)

# Model eğitimi ve backtesting (XGBoost)
model_xgb = train_model(data, model_type='xgboost')
mae_xgb, r2_xgb, results_xgb = backtest_model(model_xgb, data)

# Model eğitimi ve backtesting (Linear Regression)
model_lr = train_model(data, model_type='linear')
mae_lr, r2_lr, results_lr = backtest_model(model_lr, data)

# Tahmin sonuçlarının grafiği
def plot_backtest_results(results_rf, results_xgb, results_lr, title):
    plt.figure(figsize=(14, 7))
    plt.plot(results_rf.index, results_rf['Real_Close'], label='Gerçek Kapanış Fiyatı', marker='o', color='blue')
    plt.plot(results_rf.index, results_rf['Predicted_Close'], label='RandomForest Tahmin', marker='o', linestyle='--', color='green')
    plt.plot(results_xgb.index, results_xgb['Predicted_Close'], label='XGBoost Tahmin', marker='o', linestyle='--', color='red')
    plt.plot(results_lr.index, results_lr['Predicted_Close'], label='Linear Regression Tahmin', marker='o', linestyle='--', color='purple')
    plt.title(title)
    plt.xlabel('Tarih')
    plt.ylabel('Fiyat')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Kullanıcıdan tahmin süresi isteği al
def get_forecast_range():
    future_start_date = today + timedelta(days=30)
    future_end_date = future_start_date + timedelta(days=60)
    return pd.date_range(start=future_start_date, end=future_end_date, freq='B')  # İş günleri için

# Kullanıcıdan backtesting süresi seçimi al
def get_backtest_duration():
    print("Backtest süresi seçin:")
    print("1. 1 ay")
    print("2. 3 ay")
    print("3. 5 ay")
    choice = input("Seçiminiz (1/2/3): ")
    
    if choice == '1':
        test_size = 0.1
    elif choice == '2':
        test_size = 0.3
    elif choice == '3':
        test_size = 0.4
    else:
        print("Geçersiz seçim. 1 ay seçiliyor.")
        test_size = 0.1
    
    return test_size

# Kullanıcı tahmin isterse
def run_prediction(models, past_data, future_dates):
    future_predictions = {}
    
    for model_name, model in models.items():
        future_prices = predict_future_prices(model, past_data, future_dates)
        future_predictions[model_name] = future_prices
    
    # Tahminleri DataFrame olarak oluşturma
    future_data = pd.DataFrame({
        'Date': future_dates
    }).set_index('Date')
    
    for model_name, future_prices in future_predictions.items():
        future_data[f'Predicted_Close_{model_name}'] = future_prices
    
    
    
    return future_data

# Tahmin sonuçlarını tek grafikte çiz
def plot_future_predictions(future_data, title):
    plt.figure(figsize=(14, 7))
    
    for column in future_data.columns:
        plt.plot(future_data.index, future_data[column], label=column, marker='o', linestyle='--')
    
    plt.title(title)
    plt.xlabel('Tarih')
    plt.ylabel('Fiyat')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Kullanıcıdan seçim
print("Backtesting mi yoksa Tahmin mi yapmak istersiniz?")
print("1. Backtesting")
print("2. Tahmin")
choice = input("Seçiminiz (1/2): ")

if choice == '1':
    test_size = get_backtest_duration()
    
    # Model eğitimi ve backtesting
    model_rf = train_model(data, model_type='random_forest')
    model_xgb = train_model(data, model_type='xgboost')
    model_lr = train_model(data, model_type='linear')
    
    mae_rf, r2_rf, results_rf = backtest_model(model_rf, data, test_size)
    mae_xgb, r2_xgb, results_xgb = backtest_model(model_xgb, data, test_size)
    mae_lr, r2_lr, results_lr = backtest_model(model_lr, data, test_size)
    
    print(f"Random Forest Modeli Mean Absolute Error: {mae_rf}, R²: {r2_rf}")
    print(f"XGBoost Modeli Mean Absolute Error: {mae_xgb}, R²: {r2_xgb}")
    print(f"Linear Regression Modeli Mean Absolute Error: {mae_lr}, R²: {r2_lr}")
    
    plot_backtest_results(results_rf, results_xgb, results_lr, 'Backtesting Sonuçları (Tüm Modeller)')
    
elif choice == '2':
    future_dates = get_forecast_range()
    
    models = {
        'RandomForest': model_rf,
        'XGBoost': model_xgb,
        'LinearRegression': model_lr
    }
    
    future_data = run_prediction(models, data, future_dates)
    
    plot_future_predictions(future_data, 'Tahmin Sonuçları (Tüm Modeller)')