import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.layers import Input, Dense, GaussianNoise,Lambda,Dropout
from keras.models import Model
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam,SGD,RMSprop, Adamax, Adadelta
from keras import backend as K


np.random.seed(1)
tf.compat.v1.set_random_seed(3)
# bu kodda mesaj ve kanal sayısı parametresi sabit tutularak haberleşme sistemi...
# farklı optimizasyon algoritmaları ile gerçeklenecektir.

M = 16
k = np.log2(M)
n_channel = 1
print(n_channel)
R = k / n_channel

rates = [0.5, 1, 2, 4, 9]
colors = ['b','r','g','c','m']; c_ind =0

for i in rates:

    N_train = int(10000*i)
    label = np.random.randint(M, size=N_train)

    data = []
    for j in label:
        temp = np.zeros(M)
        temp[j] = 1
        data.append(temp) # One-Hot Encoding

    data = np.array(data)
    #print(data.shape)


    # Autoencoder Mimarisi

    input = Input(shape=(M,))
    encoded1 = Dense(M, activation='relu')(input)
    encoded2 = Dense(2*n_channel, activation='linear')(encoded1)
    encoded3 = Lambda(lambda x: np.sqrt(n_channel) * K.l2_normalize(x, axis=1))(encoded2)

    EbNo_train = 10**(10/10)  # coverted 10 db of EbNo
    encoded4 = GaussianNoise(np.sqrt(1 / (2 * R * EbNo_train)))(encoded3)

    decoded1 = Dense(M, activation='relu')(encoded4)
    decoded2 = Dense(M, activation='softmax')(decoded1)
    autoencoder = Model(input, decoded2)

    autoencoder.compile(optimizer=Adam(0.01), loss='categorical_crossentropy')

    print(autoencoder.summary())

    autoencoder.fit(data, data, epochs=50, batch_size=32)

    encoder = Model(input, encoded3)

    encoded_input = Input(shape=(2*n_channel,))
    deco1 = autoencoder.layers[-2](encoded_input)
    deco2 = autoencoder.layers[-1](deco1)
    decoder = Model(encoded_input, deco2)

    N_test = 10000
    test_label = np.random.randint(M, size=N_test)
    test_data = []

    for ii in test_label:
        temp = np.zeros(M)
        temp[ii] = 1
        test_data.append(temp) # One-Hot Encoding

    test_data = np.array(test_data)

    SNR_db = np.arange(-5,15,1)
    block_error_rate = [None] * len(SNR_db)

    for n in range(len(SNR_db)):
        SNR = 10.0 ** (SNR_db[n] / 10.0)
        noise_std = np.sqrt(1 / (2 * R * SNR)); noise_mean = 0      # Gauss gürültüsü ortalaması ve varyansı
        errors = 0                                                  # hatalı blok sayısı testin başında 0 olarak belirlendi.
        noise = noise_std * np.random.randn(N_test, 2*n_channel)   # Test'te kanala eklenecek gürültü oluşturuldu.
        encoded_signal = encoder.predict(test_data)                 # Mesaj, verici kısmından (encoder) geçerek boyut azaltıldı.
        final_signal = encoded_signal + noise                       # boyutu azaltılmış mesaja Gauss Gürültüsü eklendi.
        pred_final_signal = decoder.predict(final_signal)           # gürültülü mesaj, alıcı kısımdan (decoder) geçerek çıkış oluşturuldu.
        pred_output = np.argmax(pred_final_signal, axis=1)          # Tahmin edilen One-Hot Labellar decode edildi. (1 değerini alan index belirlendi.)
        errors = (pred_output != test_label)
        errors = errors.astype(int).sum()                           # Hatalı tahmin edilen blok sayısı belirlerdi
        block_error_rate[n] = errors / N_test                       # Blok hata oranı belirlendi.
        print('SNR:', SNR_db[n], 'BER:', block_error_rate[n])

    label = "Train-Test Data Rate: " + str(i)
    plt.plot(SNR_db, block_error_rate, 'bo',label=label,c=colors[c_ind]); c_ind+=1
    plt.xlabel('SNR Range')
    plt.ylabel('Block Error Rate')
    plt.yscale('log')




# with open('C:/users/osman/Documents/MATLAB/serQ3.pkl', 'rb') as fin :
#     BLER_8PSK = pickle.load(fin)
# with open('C:/users/osman/Documents/MATLAB/serQ4.pkl', 'rb') as fin :
#     BLER_16PSK = pickle.load(fin)
# with open('C:/users/osman/Documents/MATLAB/serQ5.pkl', 'rb') as fin :
#     BLER_32PSK = pickle.load(fin)
#
# plt.plot(SNR_db,BLER_8PSK,label='Theoric 8-PSK')
# plt.plot(SNR_db,BLER_16PSK,label='Theoric 16-PSK')
# plt.plot(SNR_db,BLER_32PSK,label='Theoric 32-PSK')

from scipy import special as sp
def qfunc(x):
    return 0.5-0.5*sp.erf(x/np.sqrt(2))
SNR = 10 ** (SNR_db / 10)

#BLER_8PSK = 2 * qfunc(np.sqrt(2 * np.log2(8) * SNR * (np.sin(np.pi/8)**2)))
BLER_16PSK = 2 * qfunc(np.sqrt(2 * np.log2(16) * SNR * (np.sin(np.pi/16)**2)))
#BLER_32PSK = 2 * qfunc(np.sqrt(2 * np.log2(32) * SNR * (np.sin(np.pi/32)**2)))

#plt.plot(SNR_db,BLER_8PSK,label='Theoric 8-PSK')
plt.plot(SNR_db,BLER_16PSK,label='Theoric 16-PSK')
#plt.plot(SNR_db,BLER_32PSK,label='Theoric 32-PSK')

plt.legend()
plt.show()