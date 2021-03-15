from os.path import split
from mastml.mastml import Mastml
from mastml.datasets import LocalDatasets, SklearnDatasets, FoundryDatasets, FigshareDatasets, DataUtilities, DataCleaning
from mastml.preprocessing import SklearnPreprocessor, MeanStdevScaler
from mastml.models import SklearnModel
from mastml.data_splitters import SklearnDataSplitter, NoSplit, LeaveOutTwinCV
from mastml.plots import Histogram, Scatter
from mastml.feature_selectors import EnsembleModelFeatureSelector, NoSelect
from mastml.feature_generators import ElementalFeatureGenerator
import sklearn

# SETUP
target = 'E_regression.1'
extra_columns = ['E_regression', 'Material compositions 1', 'Material compositions 2', 'Hop activation barrier']
d = LocalDatasets(file_path='MAST-ML/figshare_7418492/All_Model_Data.xlsx',
                  target=target,
                  extra_columns=extra_columns,
                  group_column='Material compositions 1',
                  testdata_columns=None,
                  as_frame=True)
data_dict = d.load_data()

X = data_dict['X']
y = data_dict['y']
X_extra = data_dict['X_extra']
groups = data_dict['groups']
X_testdata = data_dict['X_testdata']

mastml = Mastml(savepath='results/test_output')
savepath = mastml.get_savepath
mastml_metadata = mastml.get_mastml_metadata

cleaner = DataCleaning()
X, y = cleaner.evaluate(X=X,
                        y=y,
                        method='imputation',
                        strategy='mean',
                        savepath=savepath)

# Preprocess the cleaned data to be normalized
preprocessor = MeanStdevScaler(mean=0, stdev=1)
X = preprocessor.fit_transform(X)

# Define two models and two feature selector types to perform

model1 = SklearnModel(model='KernelRidge', kernel='rbf')
# model2 = SklearnModel(model='LinearRegression')
# models = [model1, model2]
models = [model1]

selector1 = NoSelect()
# selector2 = EnsembleModelFeatureSelector(model=SklearnModel(model='RandomForestRegressor'), k_features=10)
# selectors = [selector1, selector2]
selectors = [selector1]

# Define and run the case where leave out twins CV is performed

# test auto threshold
# splitter = LeaveOutTwinCV(threshold=0, ceiling=.1)

# splitter = LeaveOutTwinCV(threshold=0, ceiling=.1, debug=True)
splitter = NoSplit()

splitter.evaluate(X=X,
                  y=y,
                  models=models,
                  selectors=selectors,
                  savepath=savepath)

# print(f"n splits is {splitter.get_n_splits(X=X, y=y)}")

# print("CONTROL")
# splitter = SklearnDataSplitter(splitter='KFold')
# splitter.evaluate(X=X,
#                   y=y,
#                   models=models,
#                   selectors=selectors,
#                   savepath=savepath)
