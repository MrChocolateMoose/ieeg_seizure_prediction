from scipy.io import loadmat
import pandas as pd
import psutil
import os
import re

import keras.backend as K
from keras.layers.pooling import _GlobalPooling1D

def mat_to_df(path, verify_compressed_data_integrity = False):
    mat = loadmat(path, verify_compressed_data_integrity= verify_compressed_data_integrity)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    return pd.DataFrame(ndata['data'], columns=ndata['channelIndices'][0])

def mat_to_ndarray(path):
    mat = loadmat(path)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    return ndata['data']

def get_sampling_rate(path):
    mat = loadmat(path)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    return ndata['iEEGsamplingRate'].ravel()

def get_memory_usage(prefix = ""):
    process = psutil.Process(os.getpid())
    print("%s %.2f GB" % (prefix, (process.memory_info().rss / (1024.0 ** 3))))

def natural_key(string_):
    result = [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]
    return result

# TODO: standardize with above
def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]



class GlobalL2Pooling1D(_GlobalPooling1D):
    '''Global average pooling operation for temporal data.

    # Input shape
        3D tensor with shape: `(samples, steps, features)`.

    # Output shape
        2D tensor with shape: `(samples, features)`.
    '''

    def call(self, x, mask=None):
        return K.l2_normalize(x, axis=1)