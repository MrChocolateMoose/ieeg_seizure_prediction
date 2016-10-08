import glob
import re
import os
from timeit import default_timer as timer
import ntpath
import itertools
import numpy as np

import h5py
import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

from util import *

label_to_int_map = {
    'interictal' : 0,
    'preictal' : 1
}
int_to_label_map = dict((reversed(item) for item in label_to_int_map.items()))

N_CHANNELS = 16
SAMPLING_RATE = 400
NFFT = 1024
m = 80000

class DataSignal(object):

    @staticmethod
    def save_signal(types, patient_ids, labels):
        for type, patient_id, label in itertools.product(types, patient_ids, labels):
            print("(%s) %s : %s" % (type, patient_id, label))
            DataSignal.write_patient_dfs_to_hd5(type, patient_id, label)

    @staticmethod
    def write_patient_dfs_to_hd5(type, patient_id, label):

        folder = "data/" + type + "_" + str(patient_id) + "/"

        # test observations are not labeled according to 0 and 1 so reject "1" label and accept "0" label in order not to double read
        if type == 'test' and label == 1:
            pass

        # test observations won't have their label in the name
        if type == 'test':
            folder_regex = folder + "*" + ".mat"
        # train observations will have their label in the name
        else:
            folder_regex = folder + "*" + str(label) + ".mat"

        mat_files = sorted(glob.glob(folder_regex), key=natural_key)
        print(mat_files)

        # e.g. filename => 'data/train_patient_1_interictal.hdf'
        if type == 'test':
            hdf_file = "data/" + type + "_patient_" + str(patient_id) + ".hdf"
        else:
            hdf_file = "data/" + type + "_patient_" + str(patient_id) + "_" + int_to_label_map[label] + ".hdf"

        get_memory_usage("before writing file: %s" % hdf_file)
        start = timer()

        mode = 'w'
        for mat_file in mat_files:
            key = "segment_" + os.path.splitext(ntpath.basename(mat_file))[0]
            print(key)
            # key = "segment_" + os.path.splitext(ntpath.basename(mat_file))[0][2:]
            # if type == 'train':
            #    key = key[:-2]

            # assert(400 == get_sampling_rate(mat_file))
            df = mat_to_df(mat_file)
            df.to_hdf(hdf_file, key, mode=mode, complib='blosc')

            if mode == 'w':  # overwrite on first df, then append for future dfs
                mode = 'a'

        end = timer()
        get_memory_usage("after writing file: %s" % hdf_file)
        print("creating file took: %f sec" % (end - start))


def get_hdf_keys(type, patient_id, int_label):

    label = int_to_label_map[int_label]

    if type == "train":
        hdf_file = "data/" + type + "_patient_" + str(patient_id) + "_" + label + ".hdf"
    else:
        hdf_file = "data/" + type + "_patient_" + str(patient_id) + ".hdf"

    store = pd.HDFStore(hdf_file, mode='r')
    store_keys = [(store_key, type, patient_id, label) for store_key in store]
    store_keys.sort(key=lambda x: natural_sort_key(x[0]))
    print(store_keys)
    return store_keys

def get_patient_xyf_from_hdf_key(hdf_key):

    store_key, type, patient_id, label = hdf_key

    if type == "train":
        hdf_file = "data/" + type + "_patient_" + str(patient_id) + "_" + label + ".hdf"
    else:
        hdf_file = "data/" + type + "_patient_" + str(patient_id) + ".hdf"

    df = pd.read_hdf(hdf_file, store_key, mode='r')
    get_memory_usage()

    x = df.as_matrix()
    y = label_to_int_map[label]
    f = store_key

    return (x, y, f)



class DataMFCC(object):


    @staticmethod
    def signal_to_mfcc_list(signal_channels, shouldPlot=False):
        """
        :param signal_channels: a matrix whose rows are signal values over time and whose
        columns represent those signal values for a specific measurement probe. The number of columns are N_CHANNELS.
        An example in_dimension of this method would be: [240000,16]
        :param shouldPlot: optionally plot the Mel-frequency cepstral coefficients (MFCC) for each signal matrix.
        :return: a list containing each channel transformed into a matrix of Mel-frequency cepstral coefficients (MFCC)
        values. An example out_dimension of this method would be: [16, [20,469]]
        """
        channel_mfccs = []
        for i in range(0, N_CHANNELS):
            if shouldPlot:
                plt.subplot(N_CHANNELS, 1, i + 1)
            mfccs = librosa.feature.mfcc(signal_channels[:, i], sr=SAMPLING_RATE)
            # print(mfccs.shape)

            mfccs = scale(mfccs, axis=1)
            # print(mfccs.mean(axis=1))
            # print(mfccs.var(axis=1))

            if shouldPlot:
                librosa.display.specshow(mfccs, sr=SAMPLING_RATE, x_axis='time')

            if shouldPlot:
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [sec]')

            channel_mfccs.append(mfccs.T)  # rows are time, cols are coeff

        if shouldPlot:
            plt.show()

        return channel_mfccs

    @staticmethod
    def get_patient_mfcc_label(type, patient_id, int_label):
        '''
        :param type: whether the data is training data, "train", or testing data, "test"
        :param patient_id: which patient the data refers to: 1, 2, or 3
        :param int_label: request either "interictal" or "preictal" data
        :return: TODO
        '''
        hdf_keys = get_hdf_keys(type=type, patient_id=patient_id, int_label=int_label)
        XYF_list = []
        hdf_keys_length = len(hdf_keys)
        for i, hdf_key in zip(range(hdf_keys_length), hdf_keys):
            print("[%s] Patient ID %d, Label %d: fetching key... (%d/%d)" % (
            type, patient_id, int_label, i + 1, hdf_keys_length))
            XYF_list.append(get_patient_xyf_from_hdf_key(hdf_key))
        X_signals, Y, F = map(list, zip(*XYF_list))
        mfcc_list = []
        X_signals_length = len(X_signals)
        for i, X_signal in zip(range(X_signals_length), X_signals):
            print("[%s] Patient ID %d, Label %d: Converting signal to mfcc... (%d/%d)" % (
            type, patient_id, int_label, i + 1, X_signals_length))
            mfcc = np.expand_dims(np.expand_dims(DataMFCC.signal_to_mfcc_list(X_signal), axis=0), axis=0)
            mfcc_list.append(mfcc)

        X_mfcc = np.vstack(list(itertools.chain(*mfcc_list)))

        return (X_mfcc, np.asarray(Y), np.asarray(F, dtype=object))

    @staticmethod
    def get_patient_mfcc(type, patient_id, usePandasHDF=False, forceCreate=False):

        def get_mfcc_name(int_label, suffix):
            if type == "train":
                return "%s_patient_%d_%s_%s_mfcc" % (type, patient_id, int_to_label_map[int_label], suffix)
            else:
                return "%s_patient_%d_%s_mfcc" % (type, patient_id, suffix)


        # create paths
        label_0_X_file_path = os.path.join("data", get_mfcc_name(int_label=0, suffix="X") + ".np.h5")
        label_0_Y_file_path = os.path.join("data", get_mfcc_name(int_label=0, suffix="Y") + ".np.h5")
        label_0_F_file_path = os.path.join("data", get_mfcc_name(int_label=0, suffix="F") + ".np.h5")
        # check if hdf is already made for mfcc signal & label
        if forceCreate is False and os.path.isfile(label_0_X_file_path) and os.path.isfile(label_0_Y_file_path) and os.path.isfile(label_0_F_file_path):

            if usePandasHDF:
                X_mfcc_0 = pd.read_hdf(label_0_X_file_path, get_mfcc_name(int_label=0, suffix="X")).as_matrix()
                Y_0 = pd.read_hdf(label_0_Y_file_path, get_mfcc_name(int_label=0, suffix="Y")).as_matrix()
                # TODO: F_0
            else:
                with h5py.File(label_0_X_file_path, 'r') as h5_X:
                    X_mfcc_0 = np.array(h5_X.get(get_mfcc_name(int_label=0, suffix="X")))
                with h5py.File(label_0_Y_file_path, 'r') as h5_Y:
                    Y_0 = np.array(h5_Y.get(get_mfcc_name(int_label=0, suffix="Y")))
                with h5py.File(label_0_F_file_path, 'r') as h5_F:
                    F_0 = np.array(h5_F.get(get_mfcc_name(int_label=0, suffix="F")))


        # no mfcc signal & label hdf
        else:
            X_mfcc_0, Y_0, F_0 = DataMFCC.get_patient_mfcc_label(type, patient_id, int_label=0)

            if usePandasHDF:
                X_panel4D = pd.Panel4D(X_mfcc_0)
                X_panel4D.to_hdf(label_0_X_file_path, get_mfcc_name(int_label=0, suffix="X"), mode='w', complib='blosc')
                pd.Series(Y_0).to_hdf(label_0_Y_file_path, get_mfcc_name(int_label=0, suffix="Y"), mode='w',
                                      complib='blosc')
                # TODO: F_0
            else:
                with h5py.File(label_0_X_file_path, 'w') as h5_X:
                    h5_X.create_dataset(get_mfcc_name(int_label=0, suffix="X"), data=X_mfcc_0)
                with h5py.File(label_0_Y_file_path, 'w') as h5_Y:
                    h5_Y.create_dataset(get_mfcc_name(int_label=0, suffix="Y"), data=Y_0)
                with h5py.File(label_0_F_file_path, 'w') as h5_F:
                    h5_F.create_dataset(get_mfcc_name(int_label=0, suffix="F"), data=F_0, dtype = h5py.special_dtype(vlen=str))

        if type == "test":
            return (X_mfcc_0, Y_0, F_0)

        # create paths
        label_1_X_file_path = os.path.join("data", get_mfcc_name(int_label=1, suffix="X") + ".np.h5")
        label_1_Y_file_path = os.path.join("data", get_mfcc_name(int_label=1, suffix="Y") + ".np.h5")
        label_1_F_file_path = os.path.join("data", get_mfcc_name(int_label=1, suffix="F") + ".np.h5")
        if forceCreate is False and os.path.isfile(label_1_X_file_path) and os.path.isfile(label_1_Y_file_path) and os.path.isfile(label_1_F_file_path):

            if usePandasHDF:
                X_mfcc_1 = pd.read_hdf(label_1_X_file_path, get_mfcc_name(int_label=1, suffix="X")).as_matrix()
                Y_1 = pd.read_hdf(label_1_Y_file_path, get_mfcc_name(int_label=1, suffix="Y")).as_matrix()
                # TODO: F_1
            else:
                with h5py.File(label_1_X_file_path, 'r') as h5_X:
                    X_mfcc_1 = np.array(h5_X.get(get_mfcc_name(int_label=1, suffix="X")))
                with h5py.File(label_1_Y_file_path, 'r') as h5_Y:
                    Y_1 = np.array(h5_Y.get(get_mfcc_name(int_label=1, suffix="Y")))
                with h5py.File(label_1_F_file_path, 'r') as h5_F:
                    F_1 = np.array(h5_F.get(get_mfcc_name(int_label=1, suffix="F")))

        # no mfcc signal & label hdf
        else:
            X_mfcc_1, Y_1, F_1 = DataMFCC.get_patient_mfcc_label(type, patient_id, int_label=1)

            if usePandasHDF:
                pd.Panel4D(X_mfcc_1).to_hdf(label_1_X_file_path, get_mfcc_name(int_label=1, suffix="X"), mode='w',
                                            complib='blosc', format='table')
                pd.Series(Y_1).to_hdf(label_1_Y_file_path, get_mfcc_name(int_label=1, suffix="Y"), mode='w',
                                      complib='blosc')
                # TODO: F_1
            else:
                with h5py.File(label_1_X_file_path, 'w') as h5_X:
                    h5_X.create_dataset(get_mfcc_name(int_label=1, suffix="X"), data=X_mfcc_1)
                with h5py.File(label_1_Y_file_path, 'w') as h5_Y:
                    h5_Y.create_dataset(get_mfcc_name(int_label=1, suffix="Y"), data=Y_1)
                with h5py.File(label_1_F_file_path, 'w') as h5_F:
                    h5_F.create_dataset(get_mfcc_name(int_label=1, suffix="F"), data=F_1, dtype = h5py.special_dtype(vlen=str))

        X_mfcc = np.vstack((X_mfcc_0, X_mfcc_1))
        Y = np.hstack((Y_0, Y_1))
        F = np.hstack((F_0, F_1))

        return (X_mfcc, Y, F)

    @staticmethod
    def get_patients_mfcc(type, patient_ids, usePandasHDF=False):

        X = None
        Y = None
        F = None

        for patient_id in patient_ids:
            X_patient, Y_patient, F_patient = DataMFCC.get_patient_mfcc(type, patient_id, usePandasHDF)

            X = X_patient if X is None else np.vstack((X, X_patient))
            Y = Y_patient if Y is None else np.hstack((Y, Y_patient))
            F = F_patient if F is None else np.hstack((F, F_patient))

        return (X, Y, F)

    @staticmethod
    def save_mfcc(types, patient_ids):
        for type, patient_id, label in itertools.product(types, patient_ids, labels):
            X, Y, F = DataMFCC.get_patient_mfcc(type=type, patient_id=patient_id, forceCreate=True)

if __name__ == "__main__":

    types = ["test"]
    labels= [0]
    patient_ids = [2]

    #DataSignal.save_signal(types=types, patient_ids=patient_ids, labels=labels)
    DataMFCC.save_mfcc(types=types, patient_ids=patient_ids)