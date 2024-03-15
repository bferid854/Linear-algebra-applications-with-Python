"""
Merhaba bu gün 3 (farklı) şekilde En küçük kareler
yöntemiyle Lineer Regresyon katsayılarını hesaplayacağız
"""

#1 Numpy kütüphanesinin temel araçlarını kullanarak
import numpy as np

np.random.seed(42)

# Veri Seti Oluşturma
data = {'X1': np.random.rand(100),
       'X2': np.random.rand(100),
       'X3': np.random.rand(100),
       'Y': np.random.rand(100)}

# X = Bağımsız Değişkenler Matrisi
X = np.column_stack((data['X1'], data['X2'], data['X3']))

# Y = Bağımlı Değişken Vektörü
Y = data['Y']

# ÖNEMLİ: beta_0 (Sabit Terim) hesaplamak için X maatrisine bir sütun eklemek gerekmektedir.
X = np.column_stack((np.ones(len(X)), X))

# beta = (X^T . X)^(-1) . X^T . Y

beta = np.linalg.inv(X.T @ X) @ X.T @ Y

print("Çok Değişkenli (Multiple) Lineer Regresyon Katsayıları")
print("beta_0 (sabit terim)=", beta[0]),
print("beta_1 =", beta[1]),
print("beta_2 =", beta[2]),
print("beta_3 =", beta[3])

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

#2 Fonksiyon Tanımlayarak
def en_kucuk_kareler_yontemi(X, Y):
    X = np.column_stack((np.ones(len(X)), X))
    beta = np.linalg.inv(X.T @ X) @ X.T @ Y
    return beta
np.random.seed(42)

# Veri Seti Oluşturma
data = {'X1': np.random.rand(100),
       'X2': np.random.rand(100),
       'X3': np.random.rand(100),
       'Y': np.random.rand(100)}

# X = Bağımsız Değişkenler Matrisi
X = np.column_stack((data['X1'], data['X2'], data['X3']))

# Y = Bağımlı Değişken Vektörü
Y = data['Y']

beta = en_kucuk_kareler_yontemi(X, Y)

print("Çok Değişkenli (Multiple) Lineer Regresyon Katsayıları")
print("beta_0 (sabit terim)=", beta[0]),
print("beta_1 =", beta[1]),
print("beta_2 =", beta[2]),
print("beta_3 =", beta[3])

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

#3 Scikit-learn Kütüphanesi ile Lineer Regresyon Katsayılarını Hesaplama
import pandas as pd
from sklearn.linear_model import LinearRegression

np.random.seed(42)

# Veri Seti Oluşturma
data = {'X1': np.random.rand(100),
       'X2': np.random.rand(100),
       'X3': np.random.rand(100),
       'Y': np.random.rand(100)}

# Panda DataFrame Tipine Dönüştürme
df = pd.DataFrame(data)

# X = Bağımsız Değişkenler Matrisi
X = df[['X1', 'X2', 'X3']]

# Y = Bağımlı Değişken Vektörü
Y = df['Y']

# Lineer Regresyon Modeli Oluşturma
model = LinearRegression()

# Modeli fit Etme
model.fit(X, Y)

# beta_0 = intercept
print('beta_0 =', model.intercept_)

# beta_1,2,3 = coefficients
print('beta_1 =', model.coef_[0])
print('beta_2 =', model.coef_[1])
print('beta_3 =', model.coef_[2])