class no_selection:
    '''
    A class used to skip feature selection.
    '''

    def transform(self, X):
        '''
        Select all columns
        '''

        return X

    def fit(self, X, y=None):
        return self
