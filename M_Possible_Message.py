import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.layers import Input, Dense, GaussianNoise,Lambda,Dropout
from keras.models import Model
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam,SGD,RMSprop, Adamax
from keras import backend as K


np.random.seed(1)
tf.compat.v1.set_random_seed(3)
# bu kodda aynı kanal sayısına, farklı haberleşme hızına sahip psk modülsayonları ile otokodlayıcılar karşılaştırılacaktır. (Block Error Rate)

M = [4, 8, 16, 32]
k = np.log2(M)
n_channel = 1*np.ones(4); n_channel = n_channel.astype(int)
print(n_channel)
R = k / n_channel

labels = ["Q-PSK-AE","8-PSK-AE", "16-PSK-AE", "32-PSK-AE"]
colors = ['b','r','g','c']

for i in range(len(M)):

    N_train = 20000
    label = np.random.randint(M[i], size=N_train)

    data = []
    for j in label:
        temp = np.zeros(M[i])
        temp[j] = 1
        data.append(temp) # One-Hot Encoding

    data = np.array(data)
    #print(data.shape)


    # Autoencoder Mimarisi

    input = Input(shape=(M[i],)) # Giriş Katmanı (1)
    encoded1 = Dense(M[i], activation='relu')(input) # Gizli Katman (2)
    encoded2 = Dense(2*n_channel[i], activation='linear')(encoded1) # Gizli Katman (3)
    encoded3 = Lambda(lambda x: np.sqrt(n_channel[i]) * K.l2_normalize(x, axis=1))(encoded2) # Normalizasyon Katmanı (4)

    EbNo_train = 10**(9/10)  # dB'nin gerçek Eb/No değerine dönüştürülmesi
    encoded4 = GaussianNoise(np.sqrt(1 / (2 * R[i] * EbNo_train)))(encoded3) # Gürültü Kanalı (5)

    decoded1 = Dense(M[i], activation='relu')(encoded4) # Gizli Katman (6)
    decoded2 = Dense(M[i], activation='softmax')(decoded1) # Çıkış Katmanı (7)
    autoencoder = Model(input, decoded2)
    adam = Adam(lr=0.01)
    sgd = SGD(learning_rate=0.1)
    adadelta = keras.optimizers.Adadelta(2)
    rmsprop = RMSprop()
    autoencoder.compile(optimizer=adadelta, loss='categorical_crossentropy')

    print(autoencoder.summary())

    autoencoder.fit(data, data, epochs=45, batch_size=128, validation_split=0.2) # Eğitim

    encoder = Model(input, encoded3) # Mesaj işaretinin modüle edildiği kısım-Verici-Kodlayıcı

    encoded_input = Input(shape=(2*n_channel[i],)) # Kodlanan İşaret
    deco1 = autoencoder.layers[-2](encoded_input)
    deco2 = autoencoder.layers[-1](deco1)
    decoder = Model(encoded_input, deco2) # Mesaj işaretinin demodüle edildiği kısım-Alıcı-Kod Çözücü

    N_test = 10000
    test_label = np.random.randint(M[i], size=N_test)
    test_data = []

    for ii in test_label:
        temp = np.zeros(M[i])
        temp[ii] = 1
        test_data.append(temp) # One-Hot Encoding

    test_data = np.array(test_data)

    SNR_db = np.arange(-5,15,1) # Hata oranının test edileceği işaret-gürültü aralığı
    block_error_rate = [None] * len(SNR_db)

    for n in range(len(SNR_db)):
        SNR = 10.0 ** (SNR_db[n] / 10.0)
        noise_std = np.sqrt(1 / (2 * R[i] * SNR)); noise_mean = 0      # Gauss gürültüsü ortalaması ve varyansı
        errors = 0                                                  # hatalı blok sayısı testin başında 0 olarak belirlendi.
        noise = noise_std * np.random.randn(N_test, 2*n_channel[i])   # Test'te kanala eklenecek gürültü oluşturuldu.
        encoded_signal = encoder.predict(test_data)                 # Mesaj, verici kısmından (encoder) geçerek boyut azaltıldı.
        final_signal = encoded_signal + noise                       # boyutu azaltılmış mesaja Gauss Gürültüsü eklendi.
        pred_final_signal = decoder.predict(final_signal)           # gürültülü mesaj, alıcı kısımdan (decoder) geçerek çıkış oluşturuldu.
        pred_output = np.argmax(pred_final_signal, axis=1)          # Tahmin edilen One-Hot Labellar decode edildi. (1 değerini alan index belirlendi.)
        errors = (pred_output != test_label)
        errors = errors.astype(int).sum()                           # Hatalı tahmin edilen blok sayısı belirlerdi
        block_error_rate[n] = errors / N_test                       # Blok hata oranı belirlendi.
        print('SNR:', SNR_db[n], 'BER:', block_error_rate[n])

    plt.plot(SNR_db, block_error_rate, 'bo',label=labels[i],c=colors[i])
    plt.xlabel('SNR Range')
    plt.ylabel('Block Error Rate')
    plt.yscale('log')




from scipy import special as sp
def qfunc(x):
    return 0.5-0.5*sp.erf(x/np.sqrt(2))
SNR = 10 ** (SNR_db / 10)

BLER_2PSK = 2 * qfunc(np.sqrt(2 * np.log2(2) * SNR * (np.sin(np.pi/2)**2)))
BLER_4PSK = 2 * qfunc(np.sqrt(2 * np.log2(4) * SNR * (np.sin(np.pi/4)**2)))
BLER_8PSK = 2 * qfunc(np.sqrt(2 * np.log2(8) * SNR * (np.sin(np.pi/8)**2)))
BLER_16PSK = 2 * qfunc(np.sqrt(2 * np.log2(16) * SNR * (np.sin(np.pi/16)**2)))
BLER_32PSK = 2 * qfunc(np.sqrt(2 * np.log2(32) * SNR * (np.sin(np.pi/32)**2)))


plt.plot(SNR_db,BLER_4PSK,label='Theoric Q-PSK',c='b')
plt.plot(SNR_db,BLER_8PSK,label='Theoric 8-PSK',c='r')
plt.plot(SNR_db,BLER_16PSK,label='Theoric 16-PSK',c='g')
plt.plot(SNR_db,BLER_32PSK,label='Theoric 32-PSK',c='c')


plt.legend()
plt.show()