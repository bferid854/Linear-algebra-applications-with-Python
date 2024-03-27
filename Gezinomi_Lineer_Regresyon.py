import numpy as np # Numpy kütüphanesini içe aktar
import pandas as  pd # Pandas kütüphanesini içe aktar
import datetime as dt # Datetime modülünü içe aktar
from sklearn.model_selection import train_test_split # Veriyi bölmek için train_test_split fonksiyonunu içe aktar
from sklearn.linear_model import LinearRegression # Doğrusal regresyon modelini içe aktar
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error # MSE, R2, MAE değerlendirmek metriklerini içe aktar
from sklearn.decomposition import PCA # PCA modülünü içe aktarır

# Veriyi excelden okuma
df = pd.read_excel("miuul_gezinomi.xlsx")

def explore_data (data):
    #Veri setinin ilk birkaç satrını görüntüler
    print("İlk 5 gözlem:")
    print(data.head)
    print("\n")

    #Veri seti hakkında temel bilgiler sağlar:
    print("Veri seti bilgileri: ")
    print(data.info)
    print("\n")

    #Boş gözlemleri kontrol eder
    print("Boş gözlemler: ")
    print(data.isnull().sum())
    print("\n")

    #Sayısal sütunlar için temel istatistik bilgileri sağlar
    print("Sayısal sütunlar için temel istatistikler: ")
    print(data.describe())
    print("\n")

    #Veri setinin boytunu (satır ve sütun sayısını) verir
    print("Veri setinin boyutu: ")
    print(data.shape)
    print("\n")

    #Veri setinin sütun adların listeler
    print("Değişken adları")
    print(data.columns)
    print("\n")

    #Sütunların veri tipini gösterir
    print("Değişken veri tipleri :")
    print(data.dtypes)
    print("\n")

# Eksik değerlere sahip satırları sil
df.dropna(inplace=True)

# CheckInDate'den özellikler çıkarma
df['CheckInYear'] = df['CheckInDate'].dt.year
df['CheckInMonth'] = df['CheckInDate'].dt.month
df['CheckInDay'] = df['CheckInDate'].dt.day
df['CheckInWeekday'] = df['CheckInDate'].dt.dayofweek

# Günün hafta sonu olup olmadığını kontrol et
df["IsWeekend"] = np.where(df["CheckInWeekday"] >= 5, 1, 0 )

# Tatil tarihlerini tanımlama
holiday_types = {
    "Yılbaşı": ["2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01"],
    "Ulusal Egemenlik ve Çocuk Bayramı": ["2016-04-23", "2017-04-23", "2018-04-23", "2019-04-23", "2020-04-23", "2021-04-23", "2022-04-23"],
    "Emek ve Dayanışma Günü": ["2016-05-01", "2017-05-01", "2018-05-01", "2019-05-01", "2020-05-01", "2021-05-01", "2022-05-01"],
    "Atatürk'ü Anma, Gençlik ve Spor Bayramı": ["2016-05-19", "2017-05-19", "2018-05-19", "2019-05-19", "2020-05-19", "2021-05-19", "2022-05-19"],
    "Ramazan Bayramı": [("2016-07-15", "2016-07-16", "2016-07-17"),
                        ("2017-06-25", "2017-06-26", "2017-06-27"),
                        ("2018-06-15", "2018-06-16", "2018-06-17"),
                        ("2019-06-04", "2019-06-05", "2019-06-06"),
                        ("2020-05-23", "2020-05-24", "2020-05-25"),
                        ("2021-05-13", "2021-05-14", "2021-05-15"),
                        ("2022-05-02", "2022-05-03", "2022-05-04")],
    "Zafer Bayramı": ["2016-08-30", "2017-08-30", "2018-08-30", "2019-08-30", "2020-08-30", "2021-08-30", "2022-08-30"],
    "Kurban Bayramı": [("2016-09-11", "2016-09-12", "2016-09-13", "2016-09-14", "2016-09-15"),
                       ("2017-08-30", "2017-08-31", "2017-09-01", "2017-09-02", "2017-09-03"),
                       ("2018-08-21", "2018-08-22", "2018-08-23", "2018-08-24", "2018-08-25"),
                       ("2019-08-11", "2019-08-12", "2019-08-13", "2019-08-14", "2019-08-15"),
                       ("2020-07-31", "2020-08-01", "2020-08-02", "2020-08-03", "2020-08-04"),
                       ("2021-07-20", "2021-07-21", "2021-07-22", "2021-07-23", "2021-07-24"),
                       ("2022-07-09", "2022-07-10", "2022-07-11", "2022-07-12", "2022-07-13")],
    "Cumhuriyet Bayramı": ["2016-10-29", "2017-10-29", "2018-10-29", "2019-10-29", "2020-10-29", "2021-10-29", "2022-10-29"]
}

# Veri kümesindeki tatilleri işaretleme
df['HolidayType'] = 'Weekday'
for holiday, dates in holiday_types.items():
    for date_range in dates:
        if isinstance(date_range, tuple):
            for date in date_range:
                df.loc[df['CheckInDate'] == date, 'HolidayType'] = holiday
        else:
            df.loc[df['CheckInDate'] == date_range, 'HolidayType'] = holiday

# Müşteri türü için aralıkları tanımla
bins = [-1, 7, 30, 90, df['SaleCheckInDayDiff'].max()]

# Müşteri türleri için etiketleri tanımla
labels = ['Son Dakika Rezervasyoncuları', 'Potansiyel Planlayıcılar', 'Planlayıcılar', 'Erken Rezervasyon Yapanlar']

# Rezervasyon günlerine göre müşterileri kategorize et
df['CustomerType'] = pd.cut(df['SaleCheckInDayDiff'], bins=bins, labels=labels)

# Kategorik özellikleri one-hot encode et
df = pd.get_dummies(df, columns=['ConceptName', 'SaleCityName', 'CInDay', 'Seasons', 'HolidayType', 'CustomerType'])

# Özellikleri ve hedef değişkeni hazırla
X = df.drop(['Price', 'SaleId', 'SaleDate', 'CheckInDate'], axis=1)
y = df['Price']

# PCA model oluşturma
pca = PCA(n_components=5)
# Modeli fit etme
X_new = pca.fit_transform(X)
# Düşük boyutlu temsil edilen veriyi DataFrame'e dönüştürme
X_dimensional_reduced = pd.DataFrame(X_new)

# Veriyi eğitim ve test kümelerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X_dimensional_reduced, y, test_size=0.2, random_state=42)

# Modeli başlatma ve eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# beta_0 = intercept
print('beta_0 =', model.intercept_)
# beta_i = coefficients
for i, beta in enumerate(model.coef_):
    print(f'beta_{i+1} = {beta}')

# Tahminler yapma
y_pred = model.predict(X_test)

# Değerlendirme metriklerini yazdırma
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

def en_kucuk_kareler_yontemi(X, Y):
    X = np.column_stack((np.ones(len(X)), X))  # Sabit terim için X matrisine bir sütun ekler
    beta = np.linalg.inv(X.T @ X) @ X.T @ Y    # Beta katsayılarını hesaplar
    return beta

X_dimensional_reduced = X_dimensional_reduced.astype(float)
# DataFrame'i NumPy dizisine dönüştürme
X_array = X_dimensional_reduced.values
y_array = y.values


# En küçük kareler yöntemini uygula
beta = en_kucuk_kareler_yontemi(X_array, y_array)
# Hesaplanan beta katsayılarını yazdır
for i, coef in enumerate(beta):
    print(f'beta_{i} = {coef}')



