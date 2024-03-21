# 1. Gerekli kütüphanelerin yülenmesi
import numpy as np
from tensorflow.keras.datasets import mnist   # Mnist veri setini kullanmak için
from tensorflow.keras.models import Sequential # Katmanları sıralı şekilde eklemek ve yönetmek için
from tensorflow.keras.layers import Dense   # Bağlantılı sinir ağı eklemek için
from tensorflow.keras.optimizers import SGD   # Stokastik gradyan inişi uygulamak için SGD opt. algoritmatsını  kullanacağız

# 2. Veri setinin yüklenmesi
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 3. Veri ön işleme
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

# 4. Giriş verisini düzleştirme
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape((x_test.shape[0]),-1)

# 5. Model oluşturma / Sequential fonk kullanarak sinir ağları oluşturuyoruz
model = Sequential([
    Dense(128, activation = 'relu', input_shape = (784,)), # Gizli katman, Relu aktivasyonu
    Dense(10, activation = 'softmax') # Çıkış katman, 10 sınıf için softmax aktivastyonu
])

# 6. Gradient descent optimizer'ını kullanarak model derleme
sgd = SGD(learning_rate = 0.01) # Gradient descent optimizer, öğrenme hızını belirleniyor
model.compile(optimizer = sgd, # Gradient descent optimizer kullanılıyor
loss = 'sparse_categorical_crossentropy', # Kayıp fonksiyonu
metrics = ['accuracy']) # Modelin değerlendirilmesi için metrikler

# 7. Modelin eğitimi
# epoch - tüm veri setinin model üzerinden kaç gez geçirildiğini ifade eder
# batch_size - her seferdinde kullanılacak mini bach sayısını ifade eder
#(stotastik gradyan inişi için küçük batch'ler hızlı eğitim sağlar)
# validation_data : eğitim sırasında modelin performasını değerlendirmek için kullanılacak olan
#doğrulama veri setini ifade eder. Verilen epoch sayısı boyunca modeli eğitir ve her sefer modeli değerlendirir

model.fit(x_train, y_train, batch_size =32, epochs = 10, validation_data = (x_test, y_test))
# Modelin eğitilmesi: 5 epoch boyutunda, her seferinde 32 örneklik mini batch'ler kullanarak

# 8. Modelin değerlendirilmesi
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

