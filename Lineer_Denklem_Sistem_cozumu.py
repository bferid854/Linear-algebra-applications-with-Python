import numpy as np

"""
Lineer Denklem Sistemi

2𝑥1+3𝑥2−𝑥1=1
 
4𝑥1+𝑥2+2𝑥3=−2
 
−𝑥1+2𝑥2+3𝑥3=3


# 1. Numpy-ın temel yapısını kullanarak Lineer denklem çözümü
"""
# Katsayılar matrisi - A
A = np.array([[2,3,-1],
              [4,1,2],
              [-1,2,3]])

# Sabit terimler vektörü
b = np.array([1,-2,3])

# A*X=b  (x-matris)

# A- matrisinin tersini hesaplamalıyız:
# Numpy (inverse methodu) yardımıyla ters matrisin hesaplanması:

A_ters = np.linalg.inv(A)

# A matrisinin tersiyle b matrisini çarpalım: (matrix manipulation)
X = np.matmul(A_ters,b)
print("Denklem sisteminin çözümü:")
print("x_1 = ", X[0])
print("x_2 = ", X[1])
print("x_3 = ", X[2])


#-----------------------------------------------------------------------------------------


# 2. Fonksiyon tanımlayarak Lineer denklem çözümü:

def Lineer_Denklem_Sistem_Cozumu(katsayilar,sabit_terimler):
    try:
        A = np.array(katsayilar)
        b = np.array(sabit_terimler)
        A_ters = np.linalg.inv(A)
        X = np.matmul(A_ters,b)
        return X
    except np.linalg.LinAlgError:
        return "Bu denklem sisteminin çözümü yoktur!"

katsayilar = [[2,3,-1],[4,1,2],[-1,2,3]]
sabit_terimler = [1,-2,3]

cozum = Lineer_Denklem_Sistem_Cozumu(katsayilar,sabit_terimler)

print("x_1 = ", cozum[0])
print("x_2 = ", cozum[1])
print("x_3 = ", cozum[2])


#-----------------------------------------------------------------------------------------


# 3. Numpy kütüphanesindeki hazır fonksiyonu kullanarak Lineer denklem çözümü:

katsayilar = [[2,3,-1],[4,1,2],[-1,2,3]]
sabit_terimler = [1,-2,3]

x = np.linalg.solve(katsayilar,sabit_terimler)

print("x_1 = ", x[0])
print("x_2 = ", x[1])
print("x_3 = ", x[2])
