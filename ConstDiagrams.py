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

M = 16
k = np.log2(M)
n_channel = 1
R = k / n_channel

N_train = 10000
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

EbNo_train = 10**(10/10)
encoded4 = GaussianNoise(np.sqrt(1 / (2 * R * EbNo_train)))(encoded3)

decoded1 = Dense(M, activation='relu')(encoded4)
decoded2 = Dense(M, activation='softmax')(decoded1)
autoencoder = Model(input, decoded2)
adam = Adam(lr=0.01)
sgd = SGD(learning_rate=0.2)
adadelta = keras.optimizers.Adadelta(learning_rate=2)
rmsprop = RMSprop()
autoencoder.compile(optimizer=adadelta, loss='categorical_crossentropy')

print(autoencoder.summary())

callback = keras.callbacks.EarlyStopping(monitor='val_loss',patience=10)
autoencoder.fit(data, data, epochs=50, batch_size=32, validation_split=0.1, callbacks=[callback])

encoder = Model(input, encoded3)

encoded_input = Input(shape=(2*n_channel,))
deco1 = autoencoder.layers[-2](encoded_input)
deco2 = autoencoder.layers[-1](deco1)
decoder = Model(encoded_input, deco2)

N_test = 50000
test_label = np.random.randint(M, size=N_test)
test_data = []

for ii in test_label:
    temp = np.zeros(M)
    temp[ii] = 1
    test_data.append(temp) # One-Hot Encoding

test_data = np.array(test_data)

scatter_plot = []
for i in range(0,M):
    temp = np.zeros(M)
    temp[i] = 1
    scatter_plot.append(encoder.predict(np.expand_dims(temp,axis=0)))
scatter_plot = np.array(scatter_plot)


scatter_plot = scatter_plot.reshape(M,2,1)
plt.scatter(scatter_plot[:,0],scatter_plot[:,1])
plt.axis((-2.5,2.5,-2.5,2.5))
plt.grid()
plt.show()