from scipy import signal
import matplotlib.pyplot as plt
from pylab import plot, show, subplot, specgram, imshow
import librosa
import sklearn

from util import *
from load_data import *

SAMPLING_RATE = 400
NFFT = 1024
m = 80000

hdf_keys = get_patient_store_keys('train', 1, 0)


df = read_patient(hdf_keys[0])

n = 16
for i in range(0, n):
    plt.subplot(n, 1, i + 1)
    mfccs = librosa.feature.mfcc(df.as_matrix()[:, i], sr=SAMPLING_RATE)
    #print(mfccs.shape)

    mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
    #print(mfccs.mean(axis=1))
    #print(mfccs.var(axis=1))

    librosa.display.specshow(mfccs, sr=SAMPLING_RATE, x_axis='time')

    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
plt.show()
