from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Reshape
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np

import types
import tempfile
import keras.models


import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

import sklearn.base

def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__


make_keras_picklable()

def create_classifier_model_maker(in_feats, out_classes):
    def create_classifier_model():
        # create model
        model = Sequential()
        model.add(Dense(12, input_dim=in_feats, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(out_classes, activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    return create_classifier_model



#class DNNClassifier2(KerasClassifier):
#    def __call__(self):
#        self.model = create_classifier_model()
#        return self.model
#
#    def fit(self, X, y, groups=None):
#        self.model = create_classifier_model()
#        self.label_encoder = LabelEncoder()
#        self.label_encoder.fit(y)
#        encoded_y = self.label_encoder.transform(y)
#        return KerasClassifier.predict(self, X, encoded_y)
#
#    def predict(self, X, groups=None):
#        pred = KerasClassifier.predict(self, X)
#        return np.reshape(pred, pred.shape[0])
#

class DNNClassifier(sklearn.base.ClassifierMixin):
    def __init__(self, epochs=20, batch_size=5):
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.label_encoder = OneHotEncoder()
        self.label_encoder.fit(y.reshape(-1, 1))
        encoded_y = self.label_encoder.transform(y.reshape(-1, 1)).toarray()
        print('fit y: ', encoded_y)

        build_fun = create_classifier_model_maker(X.shape[1], self.label_encoder.n_values_)
        self.sk_model = KerasClassifier(build_fn=build_fun, epochs=self.epochs, batch_size=self.batch_size, verbose=1)

        self.sk_model.fit(X, encoded_y)

    def predict(self, X):
        X = np.array(X)
        pred = self.sk_model.predict(X)
        #pred_flat = np.reshape(pred, pred.shape[:-1])
        #decoded = np.argmax(pred_flat, axis=1)
        return pred

