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

def create_classifier_model(in_feats, out_classes):
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=in_feats, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(out_classes, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



class DNNClassifier2(KerasClassifier):
    def __call__(self):
        return create_classifier_model()
    def predict(self, X, groups=None):
        pred = KerasClassifier.predict(self, X)
        return np.reshape(pred, pred.shape[0])

class DNNClassifier(sklearn.base.ClassifierMixin):
    def __init__(self):
        pass

    def __call___(self):
        print('foooooooooooooooooooooooooooo')
        return self.sk_model

    def fit(self, X, y):
        model = create_classifier_model(X.shape[0], y.shape[0])
        self.sk_model = KerasClassifier(build_fn=model, epochs=20, batch_size=5, verbose=1)

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(y)
        encoded_y = self.label_encoder.transform(y)

        self.sk_model.fit(X, encoded_y)

    def predict(self, y):
        encoded_y = self.label_encoder.transform(y)
        pred = self.sk_model.predict(encoded_y)
        return np.reshape(pred, pred.shape[0])
