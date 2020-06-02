"""
This module contains a class used to randomize the input y data, in order to create a "null model" for testing how
rigorous other machine learning model predictions are.
"""

class Randomizer():
    """
    Class which randomizes X-y pairings by shuffling the y values

    Args:

        None

    Methods:

        fit: just passes through; present to maintain scikit-learn structure

            Args:

                None

        transform: randomizes the values of a dataframe

            Args:

                df: (dataframe), a dataframe with data to be randomized

            Returns:

                (dataframe), a dataframe with randomized data

    """

    def __init__(self):
        pass

    def fit(self):
        return self

    def transform(self, df):
        return df.sample(frac=1).reset_index(drop=True)