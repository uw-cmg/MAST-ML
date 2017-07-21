#!/usr/bin/env python

class BaseCustomModel():
    """Base custom model for custom models.
        Eventually replace with sklearn BaseEstimator?
    """
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        return

    def get_params(self):
        raise NotImplementedError("get_params has not been specified for this model yet") #each model specifies its own
        return

    def fit(self, input_data, target_data):
        print("Custom model does not fit by default.")
        return self

    def predict(self, input_data):
        raise NotImplementedError("predict has not been specified for this model yet") #each model specifies its own
        return
