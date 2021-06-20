from scipy import special as sp
import numpy as np
import pickle
import matplotlib.pyplot as plt

with open('C:/users/osman/Documents/MATLAB/serQ2.pkl', 'rb') as fin :
    BLER_4PSK = pickle.load(fin)


def qfunc(x):
    return 0.5-0.5*sp.erf(x/np.sqrt(2))

M = 4
SNR_db = np.arange(-5,11,1)
SNR = 10 ** (SNR_db / 10)
BLER = 2 * qfunc(np.sqrt(2 * np.log2(M) * SNR * (np.sin(np.pi/M)**2)))
print(BLER)

plt.plot(SNR_db,BLER_4PSK,label='Matlab 4-PSK')
plt.plot(SNR_db,BLER,label='Theoric 4-PSK')

plt.xlabel('SNR Range')
plt.ylabel('Block Error Rate')
plt.yscale('log')
plt.legend()
plt.show()