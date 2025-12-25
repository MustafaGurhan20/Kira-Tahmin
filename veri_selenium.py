from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import pandas as pd
import time

options = Options()
options.add_argument("--headless")  # Tarayıcıyı gizli çalıştır
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

ilanlar = []

for sayfa in range(1, 4):  # 3 sayfa örnek
    print(f"📄 Sayfa {sayfa} çekiliyor...")
    url = f"https://www.hepsiemlak.com/istanbul-satilik?page={sayfa}"
    driver.get(url)
    time.sleep(3)  # Sayfanın yüklenmesini bekle

    ilan_kartlari = driver.find_elements(By.CLASS_NAME, "listing-item")  # ilan kartlarını bul
    print(f"🔹 {len(ilan_kartlari)} ilan bulundu.")

    for ilan in ilan_kartlari:
        try:
            baslik = ilan.find_element(By.CLASS_NAME, "card-title").text
            fiyat = ilan.find_element(By.CLASS_NAME, "list-view-price").text
            adres = ilan.find_element(By.CLASS_NAME, "card-address").text
            ilanlar.append({"Baslik": baslik, "Fiyat": fiyat, "Adres": adres})
        except Exception:
            continue

driver.quit()

df = pd.DataFrame(ilanlar)
df.to_csv("hepsiemlak_istanbul.csv", index=False, encoding="utf-8-sig")
print(f"\n🎯 Toplam {len(df)} ilan kaydedildi → hepsiemlak_istanbul.csv")
