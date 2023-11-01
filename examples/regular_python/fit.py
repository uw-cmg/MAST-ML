from mastml.preprocessing import SklearnPreprocessor
from mastml.data_splitters import NoSplit
from mastml.datasets import LocalDatasets
from mastml.models import SklearnModel
from mastml.domain import Domain

target = 'E_regression_shift'
extra_columns = ['mat', 'group']

d = LocalDatasets(
                  file_path='./diffusion.csv',
                  target=target,
                  extra_columns=extra_columns,
                  testdata_columns=None,
                  as_frame=True
                  )

# Load the data with the load_data() method
data_dict = d.load_data()

# Let's assign each data object to its respective name
X = data_dict['X']
y = data_dict['y']
X_extra = data_dict['X_extra']
groups = data_dict['groups']
X_testdata = data_dict['X_testdata']

metrics = [
           'r2_score',
           'mean_absolute_error',
           'root_mean_squared_error',
           'rmse_over_stdev',
           ]

preprocessor = SklearnPreprocessor(
                                   preprocessor='StandardScaler',
                                   as_frame=True,
                                   )

model = SklearnModel(model='RandomForestRegressor')

splitter = NoSplit()

# Domain with MADML
params = {'n_repeats': 2}
domain = ('madml', params)

splitter.evaluate(
                  X=X,
                  y=y,
                  models=[model],
                  preprocessor=preprocessor,
                  metrics=metrics,
                  plots=['Scatter', 'Histogram'],
                  X_extra=X_extra,
                  verbosity=3,
                  domain=[domain],
                  )
