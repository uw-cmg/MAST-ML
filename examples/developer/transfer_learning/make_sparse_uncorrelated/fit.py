from mastml.data_splitters import SklearnDataSplitter, NoSplit
from mastml.models import SklearnModel, SourceNN, Transfer
from mastml.preprocessing import SklearnPreprocessor
from sklearn.model_selection import train_test_split
from mastml.datasets import LocalDatasets
from sklearn import datasets
from torch import nn, relu

import pandas as pd
import os


class ExampleNet(nn.Module):

    def __init__(self):

        super(ExampleNet, self).__init__()

        self.fc1 = nn.LazyLinear(24)
        self.fc2 = nn.LazyLinear(12)
        self.fc3 = nn.LazyLinear(6)
        self.fc4 = nn.LazyLinear(1)

    def forward(self, x):

        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        x = self.fc4(x)

        return x


seed = 42
nn_params = {
             'n_epochs': 1000,
             'batch_size': 32,
             'lr': 1e-4,
             'patience': 200,
             'pick': 'last',
             }
metrics = [
           'r2_score',
           'mean_absolute_error',
           'root_mean_squared_error',
           'rmse_over_stdev',
           ]

# Make regression (Would load with dataloader but this is just an example)
X, y = datasets.make_sparse_uncorrelated(
                                         n_samples=1000,
                                         random_state=seed,
                                         )

# MAST-ML compatability
X = pd.DataFrame(X)
y = pd.DataFrame(y)

# Split data into source and target domains
splits = train_test_split(
                          X,
                          y,
                          train_size=0.8,
                          random_state=0,
                          )
X_source, X_target, y_source, y_target = splits

# The target domain has same features but slightly different target
y_target = 5*y_target+2

preprocessor = SklearnPreprocessor(
                                   preprocessor='StandardScaler',
                                   as_frame=True,
                                   )

source_model = ExampleNet()  # Source NN architecture

# Create object
source_model = SourceNN(
                        source_model,
                        nn_params,
                        val_size=0.2,  # Split from whole data
                        test_size=0.5,  # Split from remaining validation
                        )

# Fit source domain model (full fit)
splitter = NoSplit()
splitter.evaluate(
                  X=X_source,
                  y=y_source,
                  models=[source_model],
                  preprocessor=preprocessor,
                  metrics=metrics,
                  plots=['Scatter', 'Histogram'],
                  verbosity=3,
                  )

os.system('rm -rf source')
os.system('mv SourceNN* source')

# The full fit source model is now in:
source_loc = './source/split_0/source/model.pth'

# Change splitter type for assessment
splitter = SklearnDataSplitter(splitter='RepeatedKFold', n_repeats=5)

# Transfer to target domain by retraining model (zero layers frozen)
freeze_model = Transfer(
                        source_loc,
                        nn_params,
                        freeze_n_layers=0,
                        val_size=0.2,  # Split from whole data
                        test_size=0.5,  # Split from remaining validation
                        )

splitter.evaluate(
                  X=X_target,
                  y=y_target,
                  models=[freeze_model],
                  preprocessor=preprocessor,
                  metrics=metrics,
                  plots=['Scatter', 'Histogram'],
                  verbosity=3,
                  )

os.system('rm -rf retrain')
os.system('mv Transfer* retrain')

# Transfer to target domain by freezing layers
freeze_model = Transfer(
                        source_loc,
                        nn_params,
                        freeze_n_layers=3,
                        val_size=0.2,  # Split from whole data
                        test_size=0.5,  # Split from remaining validation
                        )

splitter.evaluate(
                  X=X_target,
                  y=y_target,
                  models=[freeze_model],
                  preprocessor=preprocessor,
                  metrics=metrics,
                  plots=['Scatter', 'Histogram'],
                  verbosity=3,
                  )

os.system('rm -rf frozen')
os.system('mv Transfer* frozen')

# Use features from last hidden layer for another model
cover_model = SklearnModel(model='RandomForestRegressor')  # Target domain
cover_model = Transfer(
                       source_loc,
                       cover_model=cover_model,  # Use last layer as features for cover_model
                       )

splitter.evaluate(
                  X=X_target,
                  y=y_target,
                  models=[cover_model],
                  preprocessor=preprocessor,
                  metrics=metrics,
                  plots=['Scatter', 'Histogram'],
                  verbosity=3,
                  )

os.system('rm -rf cover')
os.system('mv Transfer* cover')

# Generate a random forest regressor using the original features for comparison
alone_model = SklearnModel(model='RandomForestRegressor')  # Target domain
splitter.evaluate(
                  X=X_target,
                  y=y_target,
                  models=[alone_model],
                  preprocessor=preprocessor,
                  metrics=metrics,
                  plots=['Scatter', 'Histogram'],
                  verbosity=3,
                  )

os.system('rm -rf original')
os.system('mv Random* original')
