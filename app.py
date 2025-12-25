import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# 1️⃣ Modeli yükle
# -----------------------------
model = joblib.load("real_estate_model.pkl")
st.title("🏠 Emlak Fiyat Tahmin Uygulaması (CSV Kaydetme & Analiz)")

# CSV dosya yolu
csv_file = "tahminler.csv"

# -----------------------------
# 2️⃣ Kullanıcıdan veri al
# -----------------------------
area_m2 = st.number_input("Metrekare", min_value=10, max_value=500, value=100)
rooms = st.selectbox("Oda Sayısı", [1, 2, 3, 4, 5])
age = st.number_input("Bina Yaşı", min_value=0, max_value=100, value=10)
floor = st.number_input("Kat", min_value=0, max_value=30, value=1)
building_type = st.selectbox("Bina Tipi", ['apartment', 'detached', 'duplex', 'studio'])
district = st.selectbox("Semt", ['A', 'B', 'C', 'D', 'E'])

# -----------------------------
# 3️⃣ Tahmin butonu
# -----------------------------
if st.button("Tahmin Et ve Kaydet"):
    df_input = pd.DataFrame({
        'area_m2': [area_m2],
        'rooms': [rooms],
        'age': [age],
        'floor': [floor],
        'building_type': [building_type],
        'district': [district]
    })
    price = model.predict(df_input)[0]
    st.success(f"🏷️ Tahmin edilen fiyat: {price:,.0f} TL")

    # Tahmini CSV’ye kaydet
    df_input['predicted_price'] = price
    if os.path.exists(csv_file):
        df_input.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df_input.to_csv(csv_file, index=False)

    st.info(f"Tahmin CSV’ye kaydedildi: {csv_file}")

# -----------------------------
# 4️⃣ Toplu analiz grafikleri
# -----------------------------
if os.path.exists(csv_file):
    st.subheader("📊 Toplu Tahmin Analizi")
    df_all = pd.read_csv(csv_file)

    st.write("Son tahminler:")
    st.dataframe(df_all.tail(10))  # Son 10 tahmin

    # Semtlere göre ortalama fiyat
    st.write("💰 Semtlere göre ortalama tahmin fiyatı")
    avg_price_district = df_all.groupby('district')['predicted_price'].mean()
    st.bar_chart(avg_price_district)

    # Oda sayısına göre fiyat dağılımı
    st.write("🛏️ Oda sayısına göre tahmin fiyatları")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x='rooms', y='predicted_price', data=df_all, ax=ax)
    st.pyplot(fig)
