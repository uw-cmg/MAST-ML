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

def create_classifier_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=20, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(7, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

class DNNClassifier(KerasClassifier):
    def __call__(self):
        return create_classifier_model()
    def predict(self, X, groups=None):
        pred = KerasClassifier.predict(self, X)
        return np.reshape(pred, pred.shape[0])

