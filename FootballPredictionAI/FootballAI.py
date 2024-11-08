import sqlite3
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# SQLite veritabanına bağlan
conn = sqlite3.connect('champions_league.db')
cursor = conn.cursor()

# Verileri çek
cursor.execute("SELECT * FROM matches_updated")
data = cursor.fetchall()

# Veri seti oluşturma
data = np.array(data)

# Takım adlarını sayısal değerlere dönüştür
home_teams = data[:, 0]
away_teams = data[:, 1]

# Etiket kodlayıcı oluştur ve takım adlarını kodla
label_encoder = LabelEncoder()
all_teams = np.concatenate((home_teams, away_teams))
label_encoder.fit(all_teams)

home_teams_encoded = label_encoder.transform(home_teams)
away_teams_encoded = label_encoder.transform(away_teams)

# Yeni veri seti oluştur
X = np.column_stack((home_teams_encoded, away_teams_encoded, data[:, 3].astype(int), data[:, 5].astype(int)))
y = data[:, 2].astype(int)

# Veri setinin eğitim ve test kümelerine bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelin tanımlanması
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 sınıf: Kazanma, Kaybetme, Beraberlik
])

# Modelin derlenmesi
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# EarlyStopping geri araması
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # İzlenecek metrik
    patience=10,         # İyileşme olmadan kaç epoch beklenmeli
    restore_best_weights=True  # En iyi ağırlıkları geri yükle
)

# Modelin eğitilmesi
history = model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1,
                    validation_split=0.2, callbacks=[early_stopping])

# Modelin test veri seti üzerinde değerlendirilmesi
loss, accuracy = model.evaluate(X_test, y_test)
print("Test loss:", loss)
print("Test accuracy:", accuracy)

# Yeni bir maç için tahmin yapılması
new_match_home_team = 'Real Madrid'
new_match_away_team = 'Dortmund'
new_match_must_win = 0  # Kazanma zorunluluğu var

# Yeni maçın takım adlarını kodla
new_match_home_team_encoded = label_encoder.transform([new_match_home_team])
new_match_away_team_encoded = label_encoder.transform([new_match_away_team])

# Yeni maç verilerini birleştir
new_match = np.array([[new_match_home_team_encoded[0], new_match_away_team_encoded[0], 0, new_match_must_win]])

# Tahmin yap
prediction = model.predict(new_match)
predicted_class = np.argmax(prediction, axis=1)

class_mapping = {0: 'Kaybetme', 1: 'Kazanma', 2: 'Beraberlik'}

if predicted_class == 2:  # Beraberlik durumu
    if new_match_must_win == 1:
        # Penaltı durumu için kazanan takım belirleniyor
        new_match_penalty = np.array([[new_match_home_team_encoded[0], new_match_away_team_encoded[0], 1, new_match_must_win]])
        penalty_prediction = model.predict(new_match_penalty)
        penalty_winner = np.argmax(penalty_prediction, axis=1)
        if penalty_winner == 0:
            print("Yeni maç için tahmin: Beraberlik ve penaltılarda Kaybetme")
        else:
            print("Yeni maç için tahmin: Beraberlik ve penaltılarda Kazanma")
else:
    print("Yeni maç için tahmin:", class_mapping[predicted_class[0]])

# Bağlantıyı kapat
conn.close()
