# pipeline.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# -----------------------------
# 1️⃣ Sentetik veri oluştur
# -----------------------------
def generate_real_estate_data(n=1500, seed=42):
    rng = np.random.RandomState(seed)
    area = rng.normal(100, 30, n).clip(20, 400)
    rooms = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.1, 0.3, 0.4, 0.15, 0.05])
    age = rng.randint(0, 50, n)
    floor = rng.randint(0, 15, n)
    building_type = rng.choice(['apartment', 'detached', 'duplex', 'studio'], n)
    district = rng.choice(['A', 'B', 'C', 'D', 'E'], n)

    base_price_per_m2 = {'A': 12000, 'B': 9000, 'C': 7000, 'D': 5000, 'E': 3000}
    price = []
    for a, r, ag, bt, d in zip(area, rooms, age, building_type, district):
        p = a * base_price_per_m2[d]
        p += (r - 1) * 20000
        p -= ag * 500
        if bt == 'duplex': p *= 1.1
        if bt == 'detached': p *= 1.15
        noise = rng.normal(0, 50000)
        price.append(max(30000, p + noise))

    df = pd.DataFrame({
        'area_m2': area,
        'rooms': rooms,
        'age': age,
        'floor': floor,
        'building_type': building_type,
        'district': district,
        'price': price
    })
    return df


df = generate_real_estate_data()
print("Veri örneği:")
print(df.head())

# -----------------------------
# 2️⃣ Feature / Label ayır
# -----------------------------
X = df.drop(columns=['price'])
y = df['price']

num_features = ['area_m2', 'rooms', 'age', 'floor']
cat_features = ['building_type', 'district']

# -----------------------------
# 3️⃣ Pipeline oluştur
# -----------------------------
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)

])

pipeline = Pipeline([
    ('pre', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# -----------------------------
# 4️⃣ Train/Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# -----------------------------
# 5️⃣ Tahmin ve değerlendirme
# -----------------------------
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMAE: {mae:,.0f}")
print(f"R2: {r2:.3f}")

# -----------------------------
# 6️⃣ Tahmin vs Gerçek görselleştirme
# -----------------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel("Gerçek Fiyat")
plt.ylabel("Tahmin Edilen Fiyat")
plt.title("Gerçek vs Tahmin")
plt.show()

# -----------------------------
# 7️⃣ Modeli kaydet
# -----------------------------
joblib.dump(pipeline, "real_estate_model.pkl")
print("Model kaydedildi: real_estate_model.pkl")

# -----------------------------
# 8️⃣ Kullanıcıdan veri alıp tahmin (opsiyonel)
# -----------------------------
import joblib
model = joblib.load("real_estate_model.pkl")

# Örnek manuel tahmin
example = {
    'area_m2': [120],
    'rooms': [3],
    'age': [10],
    'floor': [2],
    'building_type': ['apartment'],
    'district': ['B']
}

import pandas as pd
df_example = pd.DataFrame(example)
predicted_price = model.predict(df_example)
print("Tahmin edilen fiyat:", round(predicted_price[0], 0))

# -----------------------------
# 9️⃣ Feature importance görselleştirme
# -----------------------------
import matplotlib.pyplot as plt
import seaborn as sns

model_rf = model.named_steps['model']
preprocessor = model.named_steps['pre']

# Kategorik sütunlar one-hot oldu, isimlerini al
ohe = preprocessor.named_transformers_['cat']
cat_cols = ohe.get_feature_names_out(['building_type','district'])
all_cols = ['area_m2','rooms','age','floor'] + list(cat_cols)

importances = pd.Series(model_rf.feature_importances_, index=all_cols)
importances.sort_values().plot(kind='barh', figsize=(8,6))
plt.title("Feature Importance (Özellik Önemleri)")
plt.show()
