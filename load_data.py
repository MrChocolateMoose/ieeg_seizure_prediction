import glob
import re
import os
from timeit import default_timer as timer
import ntpath
import itertools

from util import *

label_to_int_map = {
    'interictal' : 0,
    'preictal' : 1
}
int_to_label_map = dict((reversed(item) for item in label_to_int_map.items()))

def get_patient_store_keys(type, patient_id, int_label):

    label = int_to_label_map[int_label]

    hdf_file = "data/" + type + "_patient_" + str(patient_id) + "_" + label + ".hdf"

    store = pd.HDFStore(hdf_file, mode='r')
    store_keys = [(store_key, type, patient_id, label) for store_key in store]
    store_keys.sort(key=lambda x: natural_sort_key(x[0]))
    print(store_keys)
    return store_keys

def read_patient(hdf_key):

    store_key, type, patient_id, label = hdf_key

    hdf_file = "data/" + type + "_patient_" + str(patient_id) + "_" + label + ".hdf"

    get_memory_usage("before patient read")
    df = pd.read_hdf(hdf_file, store_key, mode='r')
    get_memory_usage("after patient read")
    return df

def write_patient_dfs_to_hd5(type, patient_id, label):

    folder = "data/" + type + "_" + str(patient_id) + "/"

    mat_files = sorted(glob.glob(folder + "*" + str(label) + ".mat"), key=natural_key)
    print(mat_files)

    # e.g. filename => 'data/train_patient_1_interictal.hdf'
    hdf_file = "data/" + type + "_patient_" + str(patient_id) + "_" + int_to_label_map[label] + ".hdf"


    get_memory_usage("before writing file: %s" % hdf_file)
    start = timer()

    mode = 'w'
    for mat_file in mat_files:
        key = "segment_" + os.path.splitext(ntpath.basename(mat_file))[0][2]
        if type == 'train':
            key = key[:-2]

        print(key)

        #assert(400 == get_sampling_rate(mat_file))
        df = mat_to_df(mat_file)
        df.to_hdf(hdf_file, key, mode=mode, complib='blosc')

        if mode == 'w ': # overwrite on first df, then append for future dfs
            mode = 'a'

    end = timer()
    get_memory_usage("after writing file: %s" % hdf_file)
    print("creating file took: %f sec" % end - start)


if __name__ == "__main__":
    patient_ids = [1, 2, 3]
    labels = [0]
    type = ['train', 'test']

    for type, patient_id, label in itertools.product(type, patient_ids, labels):
        print("(%s) %s : %s" % (type, patient_id, label))
        write_patient_dfs_to_hd5(type, patient_id, label)