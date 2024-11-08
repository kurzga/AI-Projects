import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import warnings
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore", category=UserWarning)

# K-Katlı Çapraz Doğrulama Fonksiyonu
def cross_val_model(data, n_splits=5):
    X = data[['Open', 'High', 'Low', 'Volume', 'MA_20', 'MA_50', 'RSI', 'BB_upper', 'BB_lower']]
    y = data['Close']
    model = LinearRegression()
    
    kf = KFold(n_splits=n_splits, shuffle=False)
    mae_scores = []
    r2_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        mae_scores.append(mae)
        r2_scores.append(r2)

    return np.mean(mae_scores), np.mean(r2_scores)

# Eğitim ve Test Performansını Karşılaştırma
def evaluate_model(data):
    X = data[['Open', 'High', 'Low', 'Volume', 'MA_20', 'MA_50', 'RSI', 'BB_upper', 'BB_lower']]
    y = data['Close']

    # Eğitim ve Test Seti
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Model Eğitimi
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Eğitim Performansı
    train_predictions = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, train_predictions)
    train_r2 = r2_score(y_train, train_predictions)

    # Test Performansı
    test_predictions = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_predictions)
    test_r2 = r2_score(y_test, test_predictions)

    return (train_mae, train_r2), (test_mae, test_r2)

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
def train_model(data):
    X = data[['Open', 'High', 'Low', 'Volume', 'MA_20', 'MA_50', 'RSI', 'BB_upper', 'BB_lower']]
    y = data['Close']
    
    model = LinearRegression()
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

# Test Fonksiyonu
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
    
    # Doğru tahmin yüzdesi hesaplama
    tolerance = 0.01  # %1 tolerans
    accuracy = np.mean(np.abs(predictions - test_y) / test_y < tolerance) * 100  # Yüzde hesaplama
    
    results = pd.DataFrame({
        'Date': test_data.index,
        'Real_Close': test_y,
        'Predicted_Close': predictions
    }).set_index('Date')
    
    return mae, r2, accuracy, results

def plot_backtest_results(X, y, model, title, real_color='blue', pred_color='purple', real_marker='o', pred_marker='d'):
    # Out-of-Sample Testi
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model.fit(X_train, y_train)
    
    # Test seti için tahminler
    y_pred_test = model.predict(X_test)
    
    # Performans değerlendirmesi
    mae_test = mean_absolute_error(y_test, y_pred_test)
    print(f"Out-of-Sample MAE: {mae_test}")
    
    # Naive Model Benchmark
    y_naive = y.shift(1)  # Bir önceki günü kaydırarak naive model oluşturuyoruz
    y_naive = y_naive.loc[y_naive.index.isin(y.index)]  # Aynı tarihleri koruma
    y_naive = y_naive.dropna()  # NaN değerleri kaldır
    
    # Gerçek y değerlerini aynı tarihlerle kısıtlayalım
    y_real = y.loc[y_naive.index]  # Naive modelin indeksine göre gerçek değerleri al
    mae_naive = mean_absolute_error(y_real, y_naive)
    print(f"Naive Model MAE: {mae_naive}")
    
    # Statsmodels OLS model ile güven aralıkları
    X_train_with_const = sm.add_constant(X_train)
    ols_model = sm.OLS(y_train, X_train_with_const).fit()
    
    # Out-of-sample test tahminleri ve güven aralıkları
    X_test_with_const = sm.add_constant(X_test)
    predictions = ols_model.get_prediction(X_test_with_const)
    pred_summary = predictions.summary_frame(alpha=0.05)
    
    # Güven aralığı çizimi
    y_pred_ci_lower = pred_summary['mean_ci_lower']
    y_pred_ci_upper = pred_summary['mean_ci_upper']

    # Grafik çizimi
    plt.figure(figsize=(14, 7))
    plt.plot(X_test.index, y_test, label='Gerçek Kapanış Fiyatı', marker=real_marker, color=real_color)
    plt.plot(X_test.index, y_pred_test, label='Linear Regression Tahmin', marker=pred_marker, linestyle='--', color=pred_color)
    plt.fill_between(X_test.index, y_pred_ci_lower, y_pred_ci_upper, color='gray', alpha=0.3, label='Güven Aralığı (%95)')
    plt.plot(y_naive.index, y_naive, label='Naive Model Tahmin', linestyle=':', color='green')
    plt.title(title, fontsize=16)
    plt.xlabel('Tarih', fontsize=12)
    plt.ylabel('Fiyat', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Kullanıcıdan tahmin süresi isteği al
def get_forecast_range():
    future_start_date = today
    future_end_date = future_start_date + timedelta(days=30)
    return pd.date_range(start=future_start_date, end=future_end_date, freq='B')  # İş günleri için

# Kullanıcı tahmin isterse
def run_prediction(model, past_data, future_dates):
    future_prices = predict_future_prices(model, past_data, future_dates)
    future_data = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': future_prices
    }).set_index('Date')
    return future_data

# Tahmin sonuçlarını çiz
def plot_future_predictions(future_data, title):
    plt.figure(figsize=(14, 7))
    plt.plot(future_data.index, future_data['Predicted_Close'], label='Tahmin Edilen Kapanış Fiyatı', marker='o', linestyle='--', color='purple')
    plt.title(title)
    plt.xlabel('Tarih')
    plt.ylabel('Fiyat')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Tarih aralıkları
today = datetime.now()
start_date = (today - timedelta(days=730)).strftime('%Y-%m-%d')  # 2 yıllık veri
end_date = today.strftime('%Y-%m-%d')

# Gerçek borsa verileri
ticker = 'TUPRS.IS'
data = get_stock_data(ticker, start_date, end_date)

# Model eğitimi ve backtest
model_lr = train_model(data)
mae_lr, r2_lr, accuracy_lr, results_lr = backtest_model(model_lr, data)


print("Backtesting mi yoksa Tahmin mi yapmak istersiniz?")
print("1. Backtesting")
print("2. Tahmin")
choice = input("Seçiminiz (1/2): ")

if choice == '1':
    X = data[['Open', 'High', 'Low', 'Volume', 'MA_20', 'MA_50', 'RSI', 'BB_upper', 'BB_lower']]
    y = data['Close']
    
    # K-Katlı Çapraz Doğrulama ile değerlendirme
    cross_val_mae, cross_val_r2 = cross_val_model(data)
    print(f"K-Katlı Çapraz Doğrulama - MAE: {cross_val_mae}, R²: {cross_val_r2}")
    # Modeli değerlendir
    (train_mae, train_r2), (test_mae, test_r2) = evaluate_model(data)
    print(f"Eğitim Seti - MAE: {train_mae}, R²: {train_r2}")
    print(f"Test Seti - MAE: {test_mae}, R²: {test_r2}")
    print(f"Backtesting Sonuçları: MAE: {mae_lr}, R²: {r2_lr}, Accuracy: {accuracy_lr:.2f}%")
    plot_backtest_results(X, y, model_lr, 'Backtesting Sonuçları (Linear Regression)')
    
elif choice == '2':
    future_dates = get_forecast_range()
    future_data = run_prediction(model_lr, data, future_dates)
    plot_future_predictions(future_data, 'Tahmin Sonuçları (Linear Regression)')
