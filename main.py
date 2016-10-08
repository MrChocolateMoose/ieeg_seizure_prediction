import itertools

import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Sequential, Model
from keras.layers.core import Dense, Merge, Dropout, Activation, Flatten, Lambda
from keras.layers import Input, Dense, Convolution1D, MaxPooling1D, merge, LSTM, Embedding, Bidirectional
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D, _GlobalPooling1D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
#from seq2seq.layers import bidirectional

from util import *
from load_data import *

from keras.utils.visualize_util import plot




def create_model(input_shape, dropout_prob = 0.25):
    input_size = np.int32(input_shape[0] * input_shape[1]) # time, freq

    inputs = []
    channel_outputs = []
    for channel_id in range(0, N_CHANNELS):
        input = Input(shape=(input_shape[0], input_shape[1],))
        inputs.append(input)

        #print("[channel %d] input_shape: %s" % (channel_id, str(input_shape)))
        x = Convolution1D(40, filter_length=4, border_mode='valid', activation='relu')(input)
        #print("[channel %d] conv1D: %s" % (channel_id, str(x.get_shape()))) # (?, 466, 40)

        x = Dropout(dropout_prob)(x)
        print("[channel %d] dropout: %s" % (channel_id, str(x.get_shape()))) # (?, 466, 40)

        x = MaxPooling1D(pool_length=4)(x)
        print("[channel %d] maxpool1D: %s" % (channel_id, str(x.get_shape()))) # (?, 116, 40)

        x = Convolution1D(40, filter_length=4, border_mode='valid', activation='relu')(x)
        print("[channel %d] conv1D: %s" % (channel_id, str(x.get_shape())))  # (?, 113, 40)

        x = Dropout(dropout_prob)(x)
        print("[channel %d] dropout: %s" % (channel_id, str(x.get_shape())))  # (?, 113, 40)

        x = MaxPooling1D(pool_length=2)(x)
        print("[channel %d] maxpool1D: %s" % (channel_id, str(x.get_shape())))  # (?, 56, 40)

        #x = LSTM(40, return_sequences=True)(x)

        #avg_x = GlobalAveragePooling1D()(x)
        #l2_x = GlobalL2Pooling1D()(x)

        #x = merge([max_x, avg_x], mode='concat', concat_axis=1)
        channel_outputs.append(x)

    x = merge(channel_outputs, mode='concat', concat_axis=1)

    max_x = GlobalMaxPooling1D()(x)
    avg_x = GlobalAveragePooling1D()(x)

    x = merge([max_x, avg_x], mode='concat', concat_axis=1)

    predictions = Dense(2, activation='softmax')(x)
    model = Model(input=inputs, output=predictions)

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def run_kaggle():
    X, Y, F = DataMFCC.get_patients_mfcc(type="train", patient_ids=[1,2,3])

    # there is no Y
    X_test, _, F_test = DataMFCC.get_patients_mfcc(type="test", patient_ids=[1,2,3])

    seed = 1337
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.40, random_state=seed)

    input_shape = (X_train.shape[2], X_train.shape[3])
    model = create_model(input_shape=input_shape)

    #plot(model, to_file='data/model.png', show_shapes=True)
    #quit()

    X_train_channels = []
    X_valid_channels = []
    X_test_channels  = []
    for channel_id in range(0, N_CHANNELS):
        X_train_channels.append(X_train[:, channel_id])
        X_valid_channels.append(X_valid[:, channel_id])
        X_test_channels.append(X_test[:, channel_id])

    Y_train_cat = to_categorical(Y_train)
    Y_valid_cat = to_categorical(Y_valid)


    model.fit(X_train_channels, Y_train_cat, validation_data=(X_valid_channels, Y_valid_cat), nb_epoch=10, batch_size=8, verbose=1)



    print("Prediction \n")
    Y_test_predict = model.predict(X_test_channels)

    Class = np.expand_dims(Y_test_predict[:, 1], 1)
    File = np.expand_dims(pd.Series(F_test, ).apply(lambda f : str(f).replace("/segment_" , "") + ".mat" ).as_matrix(), 1)

    data = np.concatenate((File, Class), axis=1)

    submission_dataframe = pd.DataFrame(data=data, columns=["File", "Class"])
    submission_dataframe.to_csv('data/submission.csv', index=False)

    #print(roc_auc_score(Y_valid_cat, Y_predict))

if __name__ == '__main__':
    run_kaggle()
