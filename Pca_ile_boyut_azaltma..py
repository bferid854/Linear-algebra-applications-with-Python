"""
PCA Yöntemi ile Boyut Azaltma Adımları
1. Veri Setinin Standartlaştırılması:
Veri setindeki tüm değişkenlerin ortalamasını sıfır yaparak ve standart sapmasını bir birim yaparak veriyi standartlaştırma adımıdır.

2. Kovaryans Matrisinin Hesaplanması:
Standartlaştırılmış veri üzerinden değişkenler arasındaki kovaryansı gösteren matrisin oluşturulmasıdır.

3. Kovaryans Matrisinin Özdeğer ve Özvektörlerinin Bulunması:
Kovaryans matrisinin özdeğerlerini ve bu özdeğerlere karşılık gelen özvektörlerini hesaplayarak değişkenlerin temel özelliklerini belirleriz.

4. Özdeğerlerin Sıralanması ve Özvektörlerin Seçimi:
Özdeğerler büyüklüklerine göre sıralanır ve istenilen boyutta (yeni boyut) en büyük özdeğerlere karşılık gelen özvektörler seçilir.

5. Yeni Değişken Matrisinin Oluşturulması:
Seçilen özvektörler yardımıyla orijinal veri setini yeni boyutlarda temsil eden bir matris oluşturulur.

6. Boyut Azaltma ve Yeni Veri Setinin Oluşturulması:
Önceki adımlarda elde edilen özvektörler kullanılarak orijinal veri seti yeni boyutlara indirgenir.
"""

import numpy as np
import pandas as pd

np.random.seed(54)

#Veri Seti Oluşturma

data = {'X1': np.random.rand(100),
       'X2': np.random.rand(100),
       'X3': np.random.rand(100),
       'X4': np.random.rand(100),
       'X5': np.random.rand(100),
       'Y': np.random.rand(100)}

#Veri setini Pandas DataFrame Yapısına dönüştürme
df = pd.DataFrame(data)

# 1. Veri Setinin Standartlaştırılması
for column in df.columns[:-1]: # 'Y' hariç tüm sütunlar üzerinde dön
    mean_value = np.mean(df[column])

# Standartlaştırılmış Veri Setindeki Bağımsız Değişkenler Matrisi
X = df.drop('Y', axis = 1)

# 2. Kovaryans Matrisinin Hesaplanması
# COV = (X^T . X) / n

kovaryans_matrisi = (X.T @ X) / len(X)

# 3. Kovaryans Matrisinin Özdeğer(Eigenvalue) ve Özvektörlerinin(Eigenvector) Bulunması
ozdegerler, ozvektorler = np.linalg.eig(kovaryans_matrisi)

# 4. Özdeğerlerin(Eigenvalue) Sıralanması ve Özvektörlerin(Eigenvector) Seçilmesi
sirali_indexler = np.argsort(ozdegerler)[::-1] # Özdeğerlerin Indexlerinin Büyükten Küçüğe Sıralanması
sirali_ozdegerler = ozdegerler[sirali_indexler] # Özdeğerlerin Büyükten Küçüğe Sıralanması (fancy index)
sirali_vektorler = ozvektorler[sirali_indexler] # Özvektörlerin Büyükten Küçüğe Sıralanması (fancy index)

# İstenilen Boyutta Principal Componentların Seçilmesi
yeni_boyut = 2
principal_components = sirali_vektorler[: yeni_boyut]

# 5. Yeni Değişken Matrisinin Oluşturulması
X_yeni = X @ principal_components.T

# 6. Boyut İndirgenmiş Veri Setinin Oluşturulması
df_boyut_indirgenmis = pd.concat([X_yeni, df['Y']], axis=1, ignore_index=True)

print(df_boyut_indirgenmis.head())


"""
Scikit-learn ile yapılması
"""
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# Veri Seti Oluşturma
data = {'X1': np.random.rand(100),
       'X2': np.random.rand(100),
       'X3': np.random.rand(100),
       'X4': np.random.rand(100),
       'X5': np.random.rand(100),
       'Y': np.random.rand(100)}

# Veri Setini Pandas DataFrame Yapısına Dönüştürme
df = pd.DataFrame(data)

# Standartlaştırılmış Veri Setindeki Bağımsız Değişkenler Matrisi
X = df.drop('Y', axis = 1)

# PCA model oluşturma
pca = PCA(n_components=2)

# Modeli fit etme
X_new = pca.fit_transform(X)

df_dimensional_reduced = pd.concat([pd.DataFrame(X_new), df['Y']], axis=1, ignore_index=True)

print(df_dimensional_reduced.head())

"""
Sonuçlarda farklılık ortaya çıka bilir çünkü gözlem birimlerinin indexlenmesinde farklılık vardır.
"""
