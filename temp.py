import pandas as pd

import numpy as np
from numpy import hstack
from numpy import zeros
from numpy import ones
from numpy.random import rand
from numpy.random import randn

import tensorflow as tf
from keras.models import Sequential
from keras import Input
from keras.layers import Dense,LSTM
from keras.layers import Reshape
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Input

import matplotlib.pyplot as plt
from matplotlib import pyplot



LENGTH_INPUT =13

#generator model
n_outputs=LENGTH_INPUT
lstm_units=200
# Sequential modeli oluşturulur. Bu tür bir model, katmanları sıralı bir şekilde ekleyerek tanımlamanıza olanak tanır.
generator = Sequential()
#Modelin giriş katmanını ekler. Bu, bir LSTM katmanına önceki katmanın çıktısını sağlamak için kullanılacaktır.
#bu kısım datasetle ilgili olduğu için tekrar güncellenebilir

generator.add(Dense(n_outputs, activation='linear'))
generator.add(Input(shape=(100,)))
#model.add(Input(shape=(20,)))

#Bir LSTM katmanını ekler. Bu katman, önceki giriş katmanından gelen zaman serileri verilerini işler.
#generator.add(LSTM(lstm_units, return_sequences=True))
#generator.add(LSTM(lstm_units))


#Dense bir çıkış katmanı ekler. Bu katman, LSTM katmanının çıktılarını alır ve n_outputs sayısındaki çıkışı üretir
generator.add(Dense(128, activation='relu'))
generator.add(Dense(256, activation='relu'))
generator.add(Dense(512, activation='relu'))


generator.add(Dense(n_outputs, activation='linear'))
#Modelin derlenmesini sağlar.
generator.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])





# discriminator model
n_inputs=LENGTH_INPUT
dense_units=[512, 256, 128]
output_units=1
discriminator = Sequential()
discriminator.add(Dense(dense_units[0], activation='relu', input_dim=n_inputs))
discriminator.add(Dense(dense_units[1], activation='relu'))
discriminator.add(Dense(dense_units[2], activation='relu'))

discriminator.add(Dense(output_units, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

   

#GAN model
discriminator.trainable = False
# connect them
gan = Sequential()
gan.add(generator)
gan.add(discriminator)
gan.compile(loss='binary_crossentropy', optimizer='adam')


#Load the dataset from the Excel file
#First load data from .xlsx file
df = pd.read_excel("C:/Users/dell/Desktop/resources/Book1.xlsx")

X_real = df.values
    


#Real data
#n_samples=20
# Rastgele n örnek seç
#indices = np.random.randint(0, X_real.shape[1], n_samples)
#X1 = X_real[ indices]
# generate class labels  burada gerçek örnekler olduğu için etiketlerin tamamı 1'dir.
#y = ones((n, 1))
#y = np.ones((n_samples, 1))
#Fonksiyon, rastgele örnekleri ve bunlara karşılık gelen etiketleri içeren bir demet döndürür
   


# generate points in latent space as input for the generator
#latent_dim = 30
#n=20
# generate points in the latent space
#x_input = randn(latent_dim * n)
# reshape into a batch of inputs for the network
#x_input = x_input.reshape(n, latent_dim)




#TRAIN
n_epochs=200
n_batch=70
sample_interval = 10

for epoch in range(n_epochs):
    # Train discriminator
    idx = np.random.randint(0, X_real.shape[0], n_batch)
    real_data = X_real[idx]
    
   # Gürültüden rastgele bir toplu işlem üret
    noise = np.random.normal(0, 1, (n_batch, 100))

    # Gürültüden sahte veri üret
    fake_data = generator.predict(noise)
    
    print(real_data.shape) 
    print(fake_data.shape)
  
    
    
    
    # Gerçek ve sahte verileri birleştir
    X = np.concatenate((real_data, fake_data))
   
    
    # Gerçek ve sahte veriler için etiketler oluştur
    y_real = np.ones((n_batch, 1))
    y_fake = np.zeros((n_batch, 1))
    y = np.concatenate((y_real, y_fake))
    
    # Ayırt edici modeli eğit
    discriminator.trainable = True
    print(type(X))
    print(type(y))
    
    
    # X ve y'yi TensorFlow tensor'larına dönüştür
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)

    # Dönüştürülmüş tensor'ları kullanarak eğitim yapma işlemlerini gerçekleştirin
    d_loss = discriminator.train_on_batch(X_tensor, y_tensor)

    

    #d_loss = discriminator.train_on_batch(X, y)
    
    # Üreteç modeli eğit
    noise = np.random.normal(0, 1, (n_batch, 100))
    y_real = np.ones((n_batch, 1))
    discriminator.trainable = False
    g_loss = gan.train_on_batch(noise, y_real)
    
    # Eğitim ilerlemesini göster
    print(f"Epoch: {epoch}, d_loss: {d_loss}, g_loss: {g_loss}")
    # Belirli aralıklarla sentetik veri örnekleri göster
    if epoch % sample_interval == 0:
       noise = np.random.normal(0, 1, (1, 100))
       fake_data = generator.predict(noise)
       print(f"Sentetik veri örneği: {fake_data}")
    # Sentetik verileri bir DataFrame nesnesine dönüştür
    df = pd.DataFrame(fake_data, columns=df.columns)

    # Sentetik verileri bir excel dosyasına kaydet
    df.to_excel("C:/Users/dell/Desktop/resources/sentetik_veri.xlsx")
