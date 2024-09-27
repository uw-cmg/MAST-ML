"""
Module for constructing models for use in MAST-ML.

SklearnModel:
    Class that wraps scikit-learn models to have MAST-ML type functionality. Providing the model name as a string
    and the keyword arguments for the model parameters will construct the model. Note that this class also supports
    construction of XGBoost models and Keras neural network models via Keras' keras.wrappers.scikit_learn.KerasRegressor
    model.

EnsembleModel:
    Class that constructs a model which is an ensemble of many base models (sometimes called weak learners). This
    class supports construction of ensembles of most scikit-learn regression models as well as ensembles of neural
    networks that are made via Keras' keras.wrappers.scikit_learn.KerasRegressor class.

CrabNetModel:
    Class that provides an implementation of PyTorch-based CrabNet regressor model based on the following work:
    Wang, A., Kauwe, S., Murdock, R., Sparks, T. "Compositionally restricted attention-based network for
    " materials property predictions", npj Computational Materials (2021) (https://www.nature.com/articles/s41524-021-00545-1)


"""

import pandas as pd
import sklearn.base
import sklearn.utils
from sklearn.ensemble import BaggingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
import inspect
from pprint import pprint
import re
import subprocess

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import train_test_split

try:
    import transfernet as trans
    import torch
except:
    print('transfernet is an optional dependency. If you want to use transfer learning pytorch NNs, do'
          ' "pip install transfernet"')

try:
    import xgboost
except:
    print('XGBoost is an optional dependency. If you want to use XGBoost models, please manually install xgboost package with '
          'pip install xgboost. If have error with finding libxgboost.dylib library, do'
          'brew install libomp. If do not have brew on your system, first do'
          ' ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)" from the Terminal')
try:
    from sklego.linear_model import LowessRegression
except:
    print('scikit-lego is an optional dependency, enabling use of the LowessRegression model. If you want to use this model, '
          'do "pip install scikit-lego"')
try:
    from lineartree import LinearTreeRegressor, LinearForestRegressor, LinearBoostingRegressor
except:
    print('linear-tree is an optional dependency, enabling use of Linear tree, forest, and boosting models. If you want'
          ' to use this model, do "pip install linear-tree"')
try:
    from gplearn.genetic import SymbolicRegressor
except:
    print('gplearn is an optional dependency, enabling the use of genetic programming SymbolicRegressor model. If you'
          ' want to use this model, do "pip install gplearn"')
try:
    from kan import *
except:
    print('pykan is an optional dependency, enabling use of Kolmogorov-Arnold Networks (KANs). If you want to use'
          ' this mode, do "pip install pykan". As of 9/27/24, need pykan==0.0.5')


class SklearnModel(BaseEstimator, TransformerMixin):
    """
    Class to wrap any sklearn estimator, and provide some new dataframe functionality

    Args:
        model: (str), string denoting the name of an sklearn estimator object, e.g. KernelRidge

        kwargs: keyword pairs of values to include for model, e.g. for KernelRidge can specify kernel, alpha, gamma values

    Methods:
        fit: method that fits the model parameters to the provided training data
            Args:
                X: (pd.DataFrame), dataframe of X features

                y: (pd.Series), series of y target data

            Returns:
                fitted model

        predict: method that evaluates model on new data to give predictions
            Args:
                X: (pd.DataFrame), dataframe of X features

                as_frame: (bool), whether to return data as pandas dataframe (else numpy array)

            Returns:
                series or array of predicted values

        help: method to output key information on class use, e.g. methods and parameters
            Args:
                None

            Returns:
                None, but outputs help to screen
    """
    def __init__(self, model, **kwargs):
        if model == 'XGBoostRegressor':
            self.model = xgboost.XGBRegressor(**kwargs)
        elif model == 'GaussianProcessRegressor':
            kernel = kwargs['kernel']
            kernel = _make_gpr_kernel(kernel_string=kernel)
            del kwargs['kernel']
            self.model = GaussianProcessRegressor(kernel=kernel, **kwargs)
        elif model == 'LowessRegression':
            self.model = LowessRegression(**kwargs)
        elif model == 'LinearTreeRegressor':
            self.model = LinearTreeRegressor(**kwargs)
        elif model == 'LinearForestRegressor':
            self.model = LinearForestRegressor(**kwargs)
        elif model == 'LinearBoostingRegressor':
            self.model = LinearBoostingRegressor(**kwargs)
        elif model == 'SymbolicRegressor':
            self.model = SymbolicRegressor(**kwargs)
        else:
            self.model = dict(sklearn.utils.all_estimators())[model](**kwargs)

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X, as_frame=True):
        if as_frame == True:
            return pd.DataFrame(self.model.predict(X), columns=['y_pred']).squeeze()
        else:
            return self.model.predict(X).ravel()

    def predict_proba(self, X):
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)

    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def help(self):
        print('Documentation for', self.model)
        pprint(dict(inspect.getmembers(self.model))['__doc__'])
        print('\n')
        print('Class methods for,', self.model)
        pprint(dict(inspect.getmembers(self.model, predicate=inspect.ismethod)))
        print('\n')
        print('Class attributes for,', self.model)
        pprint(self.model.__dict__)
        return


class KANModel(KAN):
    '''
    Implementation of Kolmogorov-Arnold Networks (KANs) from the following work:
        Liu, Z., Wang, Y., Vaidya, S., Ruehle, F. Halverson, J., Soljacic, M. Hou, T. Y., Tegmark, M.
        "KAN: Kolmogorov-Arnold Networks", arXiv (2024) (https://arxiv.org/abs/2404.19756)

    Information on input parameters taken from pykan Github: https://github.com/KindXiaoming/pykan

    Args:
        width (list of int): list of integers specifying the network architecture. For regression problems, the first number is
            equal to the number of input features, the last number is the output number of nodes (= 1), and the intermediate
            numbers determine the number of nodes in hidden layers. Default is N - 2N+1 - 1 following the KA theorem, where
            N = number of input features

        grid (int): The number of grid intervals

        k (int): The order of piecewise polynomial

        steps (int): Number of KAN training steps. Similar to epochs for MLPs

        seed (int): the input seed. Defaults to 0 so need to set new seed for each split to have random start

        savepath (str): path to save the output model

        opt (str): optimization method. Possibilities include "LBFGS", or "Adam"

        lamb (float): overall penalty strength

        lamb_entropy (float): entropy penalty strength

    Methods:

        fit: method that fits the model parameters to the provided training data
            Args:
                X: (pd.DataFrame), dataframe of X data used for model training

                y: (pd.Series), series of y target data

            Returns:
                fitted model

        predict: method that evaluates model on new data to give predictions
            Args:
                X: (pd.DataFrame), dataframe of X data used for model testing

                as_frame: (bool), whether to return data as pandas dataframe (else numpy array)

            Returns:
                series or array of predicted values
    '''

    def __init__(self, width, grid=3, k=3, steps=20, seed=None, savepath=None, opt='LBFGS', lamb=0.01, lamb_entropy=10):
        self.width = width
        self.grid = grid
        self.k = k
        self.steps = steps
        self.seed = seed
        self.savepath = savepath
        self.opt = opt
        self.lamb = lamb
        self.lamb_entropy = lamb_entropy
        super(KANModel, self).__init__(width=self.width, grid=self.grid, k=self.k)

    def fit(self, X, y):
        # Need to re-instantiate the model at each fit, otherwise data leakage will occur e.g., when doing 5fold CV
        super(KANModel, self).__init__(width=self.width, grid=self.grid, k=self.k)
        if self.seed is None:
            # get new random seed
            seed = np.random.randint(0, 1000000000, 1)[0]
            self.seed = seed

        if self.width is None:
            # set default width
            N = X.shape[1]
            self.width = [N, 2*N+1, 1]

        # Make data into torch tensors
        import torch
        dataset = {'train_input': torch.from_numpy(np.array(X)),
                   'test_input': torch.from_numpy(np.array(X)),
                   'train_label': torch.from_numpy(np.array(y).reshape(-1, 1)),
                   'test_label': torch.from_numpy(np.array(y).reshape(-1, 1))}

        self(dataset['train_input'])
        self.train(dataset, opt=self.opt, steps=self.steps, lamb=self.lamb, lamb_entropy=self.lamb_entropy)

        return

    def predict(self, X, as_frame=True):
        dataset = {'test_input': torch.from_numpy(np.array(X))}

        preds = self(dataset['test_input']).detach().numpy().ravel()

        if as_frame == True:
            return pd.DataFrame(preds, columns=['y_pred']).squeeze()
        else:
            return preds

class SourceNN:

    def __init__(
                 self,
                 source_arch,
                 nn_params,
                 val_size=None,
                 test_size=None,
                 savepath='source_model',
                 ):

        self.source_arch = source_arch  # NN Architecture
        self.nn_params = nn_params  # Fitting marameters like lr and batch size
        self.val_size = val_size
        self.test_size = test_size
        self.savepath = savepath  # The location to save results

    def fit(self, X_train, y_train):

        # Split source domain into train and validation
        if self.val_size is not None:
            splits = train_test_split(
                                      X_train,
                                      y_train,
                                      test_size=self.val_size,
                                      )
            X_train, X_val, y_train, y_val = splits

            # Split source domain validation to get test set
            if self.test_size is not None:
                splits = train_test_split(
                                          X_val,
                                          y_val,
                                          test_size=self.test_size,
                                          )
                X_val, X_test, y_val, y_test = splits

            else:
                X_test = y_test = None

        else:
            X_val = y_val = X_test = y_test = None

        # Fit the NN
        out = trans.utils.fit(
                              self.source_arch,
                              X_train,
                              y_train,
                              X_val=X_val,
                              y_val=y_val,
                              X_test=X_test,
                              y_test=y_test,
                              save_dir=self.savepath+'/source',
                              **self.nn_params,
                              )

        self.source_model = out[1]  # The fitted NN
        self.source_model = model_wrapper(self.source_model)

    def predict(self, X):
        return self.source_model.predict(X)

class Transfer:

    def __init__(
                 self,                
                 prefit_path,
                 nn_params=None,
                 val_size=None,
                 test_size=None,
                 freeze_n_layers=None,
                 cover_model=None,
                 savepath='transfer_model',
                 ):
          
        self.nn_params = nn_params  # Fitting marameters like lr and batch size
        self.val_size = val_size          
        self.test_size = test_size
        self.freeze_n_layers = freeze_n_layers
        self.cover_model = cover_model
        self.prefit_path = prefit_path
        self.savepath = savepath  # The location to save results

    def fit(self, X_train, y_train):

        # Split source domain into train and validation
        if self.val_size is not None:
            splits = train_test_split(
                                      X_train,
                                      y_train,
                                      test_size=self.val_size,
                                      )
            X_train, X_val, y_train, y_val = splits
          
            # Split source domain validation to get test set
            if self.test_size is not None:
                splits = train_test_split(
                                          X_val,
                                          y_val,
                                          test_size=self.test_size,
                                          )
                X_val, X_test, y_val, y_test = splits

            else:
                X_test = y_test = None

        else:
            X_val = y_val = X_test = y_test = None

        # Chose defalut device
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        # Load a prefit NN model
        self.source_model = torch.load(
                                       self.prefit_path,
                                       map_location=torch.device(device),
                                       )

        # Use final hidden layer of NN as features
        if self.cover_model is not None:
            self.target_model = trans.models.AppendModel(
                                                         self.source_model,
                                                         self.cover_model,
                                                         )

            self.target_model.fit(X_train, y_train)

            if all([X_test is not None, y_test is not None]):

                y_pred = self.target_model.predict(X_test)

                df = pd.DataFrame()
                df['y'] = y_test.ravel()
                df['y_pred'] = y_pred
                df['set'] = 'test'

                name = self.savepath+'/transfer_by_appending_model'
                os.makedirs(name, exist_ok=True)
                trans.plots.parity(df, os.path.join(name, 'parity'))

        # Alternatively freeze layers of NN
        elif self.freeze_n_layers is not None:

            name = '/transfer_by_freeze_{}_layers'.format(self.freeze_n_layers)
            out = trans.utils.fit(
                                  self.source_model,  # Start from original model
                                  X_train,
                                  y_train,
                                  X_val=X_val,
                                  y_val=y_val,
                                  X_test=X_test,
                                  y_test=y_test,
                                  save_dir=self.savepath+name,
                                  freeze_n_layers=self.freeze_n_layers,
                                  **self.nn_params,
                                  )

            self.target_model = out[1]
            self.target_model = model_wrapper(self.target_model)

        return self.target_model

    def predict(self, X):
        return self.target_model.predict(X)


class model_wrapper:
    '''
    Wrapper for pytorch model to include predict method
    '''

    def __init__(self, model):
        self.model = model

    def predict(self, X):

        # Chose defalut device
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        X = trans.utils.to_tensor(X, device)

        pred = self.model(X)  # If model object passed
        pred = pred.cpu().detach().view(-1).numpy()

        return pred


class HostedModel():

    def __init__(self, container_name):
        self.container_name = container_name

    def predict(self, df):

        df.to_csv('./test.csv', index=False)

        command = 'udocker --allow-root run -v '
        command += '{}:/mnt '.format(os.getcwd())
        command += self.container_name
        command += ' python3 predict.py'

        subprocess.check_output(
                                command,
                                shell=True
                                )

        df = pd.read_csv('./prediction.csv')

        os.remove('./test.csv')
        os.remove('./prediction.csv')

        return df

class EnsembleModel(BaseEstimator, TransformerMixin):
    """
    Class used to construct ensemble models with a particular number and type of weak learner (base model). The
    ensemble model is compatible with most scikit-learn regressor models and KerasRegressor models

    Args:
        model: (str), string name denoting the name of the model type to use as the base model

        n_estimators: (int), the number of base models to include in the ensemble

        kwargs: keyword arguments for the base model parameter names and values

    Methods:
        fit: method that fits the model parameters to the provided training data
            Args:
                X: (pd.DataFrame), dataframe of X features

                y: (pd.Series), series of y target data

            Returns:
                fitted model

        predict: method that evaluates model on new data to give predictions
            Args:
                X: (pd.DataFrame), dataframe of X features

                as_frame: (bool), whether to return data as pandas dataframe (else numpy array)

            Returns:
                series or array of predicted values

        get_params: method to output key model parameters
            Args:
                deep: (bool), determines the extent of information returned, default True

            Returns:
                information on model parameters
    """
    def __init__(self, model, n_estimators, **kwargs):
        super(EnsembleModel, self).__init__()
        try:
            if model == 'XGBoostRegressor':
                model = xgboost.XGBRegressor(**kwargs)
            elif model == 'GaussianProcessRegressor':
                kernel = kwargs['kernel']
                kernel = _make_gpr_kernel(kernel_string=kernel)
                del kwargs['kernel']
                model = GaussianProcessRegressor(kernel=kernel, **kwargs)
            elif model == 'KANModel':
                model = KANModel(**kwargs)
            else:
                model = dict(sklearn.utils.all_estimators())[model](**kwargs)
        except:
            print('Could not find designated model type in scikit-learn model library. Note the other supported model'
                  'type is the keras.wrappers.scikit_learn.KerasRegressor model')
        self.n_estimators = n_estimators
        self.model = BaggingRegressor(estimator=model, n_estimators=self.n_estimators)
        self.base_estimator_ = model.__class__.__name__

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X, as_frame=True):
        if as_frame == True:
            return pd.DataFrame(self.model.predict(X), columns=['y_pred']).squeeze()
        else:
            return self.model.predict(X).ravel()

    def get_params(self, deep=True):
        return self.model.get_params(deep)


class CrabNetModel(BaseEstimator, TransformerMixin):
    '''
    Implementation of PyTorch-based CrabNet regressor model based on the following work:
        Wang, A., Kauwe, S., Murdock, R., Sparks, T. "Compositionally restricted attention-based network for
        " materials property predictions", npj Computational Materials (2021) (https://www.nature.com/articles/s41524-021-00545-1)

    The code to run CrabNet was integrated into MAST-ML based on source files available in this repository: https://github.com/anthony-wang/CrabNet

    Args:
        composition_column (str): string denoting the name of an input column containing materials compositions

        epochs (int): number of training epochs

        val_frac (float): fraction of input data to use as validation (0 to 1). Note this can be set to 0 for most MAST-ML
            runs as the splitting is done outside of this model.

        drop_unary (bool): whether or not to drop compositions containing one element. Should probably be False always. If true,
            can cause some array length mismatches.

        savepath (str): path to save the data to. This is set automatically in the data_splitters routine.

    Methods:
        get_model: method that prepares the CrabNet model. Called when the CrabNetModel class is instantiated.

        fit: method that fits the model parameters to the provided training data
            Args:
                X: (pd.DataFrame), dataframe of X data, needs to contain at least a column of material compositions. Note
                    that featurization is done internally by the model.

                y: (pd.Series), series of y target data

            Returns:
                fitted model

        predict: method that evaluates model on new data to give predictions
            Args:
                X: (pd.DataFrame), dataframe of X data, needs to contain at least a column of material compositions. Note
                    that featurization is done internally by the model.

                as_frame: (bool), whether to return data as pandas dataframe (else numpy array)

            Returns:
                series or array of predicted values
    '''
    def __init__(self, composition_column, epochs, val_frac, drop_unary=False, savepath=None):
        super(CrabNetModel, self).__init__()
        self.composition_column = composition_column
        self.epochs = epochs
        self.val_frac = val_frac
        self.drop_unary = drop_unary
        self.savepath = savepath

        #self.get_model()

    def get_model(self):
        from mastml.crabnet.model import Model
        from mastml.crabnet.kingcrab import CrabNet
        from mastml.utils.get_compute_device import get_compute_device

        compute_device = get_compute_device(prefer_last=True)

        model = Model(CrabNet(compute_device=compute_device).to(compute_device),
                      model_name='crabnet_model', verbose=True, drop_unary=self.drop_unary)

        self.model = model
        return

    def fit(self, X, y):
        self.get_model()

        # Crabnet needs to read a csv for data. The data needs compositions with "formula" as column name and targets with "target" as column name
        train_data = os.path.join(self.savepath, 'train_crabnet.csv')
        val_data = os.path.join(self.savepath, 'val_crabnet.csv')

        all_data = pd.DataFrame({'formula': X[self.composition_column].values, 'target': y.values})
        if self.val_frac > 0:
            from sklearn.model_selection import train_test_split
            trains, vals = train_test_split(all_data, test_size=self.val_frac)
        else:
            trains = all_data
        trains.to_csv(train_data, index=False)
        if self.val_frac > 0:
            vals.to_csv(val_data, index=False)

        data_size = trains.shape[0]
        batch_size = 2 ** round(np.log2(data_size) - 4)
        if batch_size < 2 ** 7:
            batch_size = 2 ** 7
        if batch_size > 2 ** 12:
            batch_size = 2 ** 12

        self.batch_size = batch_size
        self.model.load_data(train_data, batch_size=self.batch_size, train=True)

        print(f'training with batchsize {self.model.batch_size} '
              f'(2**{np.log2(self.model.batch_size):0.3f})')

        if self.val_frac > 0:
            self.model.load_data(val_data, batch_size=batch_size)

        # Set the number of epochs, decide if you want a loss curve to be plotted
        self.model.fit(epochs=self.epochs, losscurve=False)

        # Save the network (saved as f"{model_name}.pth")
        self.model.save_network(path=self.savepath)
        return

    def predict(self, X, as_frame=True):
        test_data = os.path.join(self.savepath, 'test_crabnet.csv')
        data = pd.DataFrame({'formula': X[self.composition_column].values, 'target': np.zeros(shape=X.shape[0])})
        data.to_csv(test_data, index=False)

        self.model.load_data(test_data, batch_size=self.batch_size, train=False)
        output = self.model.predict(self.model.data_loader)
        preds = output[1]

        if as_frame == True:
            return pd.DataFrame(preds, columns=['y_pred']).squeeze()
        else:
            return preds.ravel()

def _make_gpr_kernel(kernel_string):
    """
    Method to transform a supplied string to a kernel object for use in GPR models

    Args:
        kernel_string: (str), a string containing the desired name of the kernel

    Return:
        kernel: sklearn.gaussian_process.kernels object

    """
    kernel_list = ['WhiteKernel', 'RBF', 'ConstantKernel', 'Matern', 'RationalQuadratic', 'ExpSineSquared', 'DotProduct']
    kernel_operators = ['+', '*', '-']
    # Parse kernel_string to identify kernel types and any kernel operations to combine kernels
    kernel_types_asstr = list()
    kernel_types_ascls = list()
    kernel_operators_used = list()

    for s in kernel_string[:]:
        if s in kernel_operators:
            kernel_operators_used.append(s)

    # Do case for single kernel, no operators
    if len(kernel_operators_used) == 0:
        kernel_types_asstr.append(kernel_string)
    else:
        # New method, using re
        unique_operators = np.unique(kernel_operators_used).tolist()
        unique_operators_asstr = '['
        for i in unique_operators:
            unique_operators_asstr += str(i)
        unique_operators_asstr += ']'
        kernel_types_asstr = re.split(unique_operators_asstr, kernel_string)

    for kernel in kernel_types_asstr:
        kernel_ = getattr(sklearn.gaussian_process.kernels, kernel)
        kernel_types_ascls.append(kernel_())

    # Case for single kernel
    if len(kernel_types_ascls) == 1:
        kernel = kernel_types_ascls[0]

    kernel_count = 0
    for i, operator in enumerate(kernel_operators_used):
        if i+1 <= len(kernel_operators_used):
            if operator == "+":
                if kernel_count == 0:
                    kernel = kernel_types_ascls[kernel_count] + kernel_types_ascls[kernel_count+1]
                else:
                    kernel += kernel_types_ascls[kernel_count+1]
            elif operator == "*":
                if kernel_count == 0:
                    kernel = kernel_types_ascls[kernel_count] * kernel_types_ascls[kernel_count+1]
                else:
                    kernel *= kernel_types_ascls[kernel_count+1]
            else:
                print('Warning: You have chosen an invalid operator to construct a composite kernel. Please choose'
                              ' either "+" or "*".')
            kernel_count += 1

    return kernel
