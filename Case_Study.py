# Gerekli Kütüphaneleri yükleyelim
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Veri setini yükleyelim ve değişkenleri ayıralım
diabets_data = load_diabetes()
X = pd.DataFrame(diabets_data.data, columns=diabets_data.feature_names)
Y = pd.Series(diabets_data.target, name = "target")

# Veri setini inceleyelim
print("Veri seti seçimi: ")
print(X.head())

# Veri setini keşf etme ve Temizleme
if X.isnull().sum().sum() == 0:
    print("Veri setinde eksik veri yok.")
else:
    print("Veri setinde eksik veriler var. Eksik verileri ortalama ile dolduruyoruz.")
    X.fillna(X.mean(), inplace=True)

# PCA ile boyut azaltma işlemi
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)
print("Boyut Azaltılmış Veri")
print(pd.DataFrame(X_pca, columns=[f"Component {i+1}" for i in range(5)]).head())

# En küçük kareler yöntemi kullanarak Lineer regresyon modeli oluşturuyoruz
X_train, X_test, y_train, y_test = train_test_split(X_pca, Y, test_size=0.2, random_state=42)
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, y_train)

# Oluşturduğumuz modelin performansını değerlendirme
y_pred = linear_reg_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Ortalama Hata Kare:{mse}")
print("Ortalama Mutlak Hata: {mae}")
print("R2 Skoru: {r2}")

#Katsayıları ve sabit'i yazdıralım
coefficients = linear_reg_model.coef_
intercept = linear_reg_model.intercept_
for i, coef in enumerate(coefficients):
    print(f"Bileşen {i+1}: {coef}")
print("\nIntercept: {intercept}")