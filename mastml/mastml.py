"""
This module contains routines to set up and manage the metadata for a MAST-ML run

Mastml:
    Class to set up directories for saving the output of a MAST-ML run, and for constructing and updating a
    metadata summary file.

"""

import os
from datetime import datetime
from collections import OrderedDict
import json
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial

class Mastml():
    """
    Main helper class to initialize mastml runs and create and manage run metadata

    Args:
        savepath: (str), string specifing the savepath name for the mastml run

        mastml_metdata: (dict), dict of mastml metadata. If none, a new dict will be created

    Methods:
        _initialize_run: initializes run by making new metadata file or updating existing one, and initializing the output directory.
            Args:
                None

            Returns:
                None

        _initialize_output: creates the output folder based on specified savepath and datetime information
            Args:
                None

            Returns:
                None

        _initialize_metadata: creates a new metadata file and saves the savepath info to it
            Args:
                None

            Returns:
                None

        _update_metadata: placeholder for updating the metadata file with new run information
            Args:
                None

            Returns:
                None

        _save_mastml_metadata: saves the metadata dict as a json file
            Args:
                None

            Returns:
                None

        get_savepath: returns the savepath
            Args:
                None

            Returns:
                string specifying the savepath of the mastml run

        get_mastml_metadata: returns the metadata file
            Args:
                None

            Returns:
                mastml metadata object (ordered dict)

    """
    def __init__(self, savepath, mastml_metadata=None):
        self.savepath = savepath
        self.mastml_metadata = mastml_metadata
        self._initialize_run()

    def _initialize_run(self):
        self._initialize_output()
        if self.mastml_metadata is None:
            self._initialize_metadata()
        #else:
        #    self._update_metadata()
        self._save_mastml_metadata()

    def _initialize_output(self):
        # Make an output folder for the run to store all data to
        if os.path.exists(self.savepath):
            try:
                os.rmdir(self.savepath)  # succeeds if empty
            except OSError:  # directory not empty
                print(f"{self.savepath} not empty. Renaming...")
                now = datetime.now()
                self.savepath = self.savepath.rstrip(os.sep)  # remove trailing slash
                self.savepath = f"{self.savepath}_{now.year:02d}_{now.month:02d}_{now.day:02d}" \
                         f"_{now.hour:02d}_{now.minute:02d}_{now.second:02d}"
        os.makedirs(self.savepath)
        return

    def _initialize_metadata(self):
        self.mastml_metadata = OrderedDict()
        self.mastml_metadata['savepath'] = self.savepath
        return

    def _update_metadata(self,
                         outerdir,
                         split_name,
                         model=None,
                         splitter=None,
                         preprocessor=None,
                         selector=None,
                         hyperopt=None,
                         train_stats=None,
                         test_stats=None,
                         leaveout_stats=None,
                         X_train=None,
                         X_test=None,
                         X_leaveout=None,
                         X_extra_train=None,
                         X_extra_test=None,
                         X_extra_leaveout=None,
                         y_train=None,
                         y_test=None,
                         y_test_domain=None,
                         y_leaveout=None,
                         y_pred_train=None,
                         y_pred=None,
                         y_pred_leaveout=None,
                         residuals_train=None,
                         residuals_test=None,
                         residuals_leaveout=None,
                         model_errors_train=None,
                         model_errors_test=None,
                         model_errors_leaveout=None,
                         model_errors_train_cal=None,
                         model_errors_test_cal=None,
                         model_errors_leaveout_cal=None,
                         dataset_stdev=None):
        # Update with new entry: (1) module, (2) class, (3) path executed, (4) paths to data used ???
        if outerdir not in self.mastml_metadata.keys():
            self.mastml_metadata[outerdir] = OrderedDict()
        if split_name not in self.mastml_metadata[outerdir].keys():
            self.mastml_metadata[outerdir][split_name] = OrderedDict()
        if split_name == 'split_outer_dir':
            self.mastml_metadata[outerdir][split_name]['splitdir'] = outerdir
        else:
            self.mastml_metadata[outerdir][split_name]['splitdir'] = split_name
        if model is not None:
            try:
                model_name = model.model.__class__.__name__
            except:
                model_name = model.__class__.__name__
            self.mastml_metadata[outerdir][split_name]['model'] = model_name

            if split_name == 'split_summary':
                self.mastml_metadata[outerdir][split_name]['model_path'] = os.path.join(os.path.join(self.savepath, outerdir), model_name+'.pkl')
            elif split_name == 'split_outer_summary':
                self.mastml_metadata[outerdir][split_name]['model_path'] = os.path.join(outerdir, model_name+'.pkl')
            else:
                self.mastml_metadata[outerdir][split_name]['model_path'] = os.path.join(os.path.join(os.path.join(self.savepath, outerdir), split_name), model_name + '.pkl')
        if splitter is not None:
            self.mastml_metadata[outerdir][split_name]['splitter'] = splitter.splitter.__class__.__name__
        if preprocessor is not None:
            self.mastml_metadata[outerdir][split_name]['preprocessor'] = preprocessor.__class__.__name__
        if selector is not None:
            self.mastml_metadata[outerdir][split_name]['selector'] = selector.__class__.__name__
        if hyperopt is not None:
            self.mastml_metadata[outerdir][split_name]['hyperopt'] = hyperopt.__class__.__name__
        if train_stats is not None:
            self.mastml_metadata[outerdir][split_name]['train_stats'] = train_stats.to_json()
        if test_stats is not None:
            self.mastml_metadata[outerdir][split_name]['test_stats'] = test_stats.to_json() #to_dict
        if leaveout_stats is not None:
            self.mastml_metadata[outerdir][split_name]['leaveout_stats'] = leaveout_stats.to_json()
        if X_train is not None:
            self.mastml_metadata[outerdir][split_name]['train_columns'] = X_train.columns.tolist()
            self.mastml_metadata[outerdir][split_name]['X_train'] = X_train.to_json()
        if X_test is not None:
            self.mastml_metadata[outerdir][split_name]['X_test'] = X_test.to_json()
        if X_leaveout is not None:
            self.mastml_metadata[outerdir][split_name]['X_leaveout'] = X_leaveout.to_json()
        if X_extra_train is not None:
            self.mastml_metadata[outerdir][split_name]['X_extra_train'] = X_extra_train.to_json()
        if X_extra_test is not None:
            self.mastml_metadata[outerdir][split_name]['X_extra_test'] = X_extra_test.to_json()
        if X_extra_leaveout is not None:
            self.mastml_metadata[outerdir][split_name]['X_extra_leaveout'] = X_extra_leaveout.to_json()
        if y_train is not None:
            self.mastml_metadata[outerdir][split_name]['y_train'] = y_train.to_json()
        if y_test is not None:
            self.mastml_metadata[outerdir][split_name]['y_test'] = y_test.to_json()
        if y_leaveout is not None:
            self.mastml_metadata[outerdir][split_name]['y_leaveout'] = y_leaveout.to_json()
        if y_pred_train is not None:
            self.mastml_metadata[outerdir][split_name]['y_pred_train'] = y_pred_train.to_json()
        if y_pred is not None:
            self.mastml_metadata[outerdir][split_name]['y_pred'] = y_pred.to_json()
        if y_pred_leaveout is not None:
            self.mastml_metadata[outerdir][split_name]['y_pred_leaveout'] = y_pred_leaveout.to_json()
        if y_test_domain is not None:
            self.mastml_metadata[outerdir][split_name]['y_test_domain'] = y_test_domain.to_json()
        if residuals_train is not None:
            self.mastml_metadata[outerdir][split_name]['residuals_train'] = residuals_train.to_json()
        if residuals_test is not None:
            self.mastml_metadata[outerdir][split_name]['residuals_test'] = residuals_test.to_json()
        if residuals_leaveout is not None:
            self.mastml_metadata[outerdir][split_name]['residuals_leaveout'] = residuals_leaveout.to_json()
        if model_errors_train is not None:
            self.mastml_metadata[outerdir][split_name]['model_errors_train'] = model_errors_train.to_json()
        if model_errors_test is not None:
            self.mastml_metadata[outerdir][split_name]['model_errors_test'] = model_errors_test.to_json()
        if model_errors_leaveout is not None:
            self.mastml_metadata[outerdir][split_name]['model_errors_leaveout'] = model_errors_leaveout.to_json()
        if model_errors_train_cal is not None:
            self.mastml_metadata[outerdir][split_name]['model_errors_train_cal'] = model_errors_train_cal.to_json()
        if model_errors_test_cal is not None:
            self.mastml_metadata[outerdir][split_name]['model_errors_test_cal'] = model_errors_test_cal.to_json()
        if model_errors_leaveout_cal is not None:
            self.mastml_metadata[outerdir][split_name]['model_errors_leaveout_cal'] = model_errors_leaveout_cal.to_json()
        if dataset_stdev is not None:
            self.mastml_metadata[outerdir][split_name]['dataset_stdev'] = dataset_stdev
        return

    def _save_mastml_metadata(self):
        with open(os.path.join(self.savepath, 'mastml_metadata.json'), 'w') as f:
            json.dump(self.mastml_metadata, f)
        return

    @property
    def get_savepath(self):
        return self.savepath

    @property
    def get_mastml_metadata(self):
        return self.mastml_metadata

def parallel(func, x, *args, **kwargs):
    '''
    Run some function in parallel.

    inputs:
        func = The function to apply.
        x = The list of items to apply function on.

    outputs:
        data = List of items returned by func.
    '''

    pool = Pool(os.cpu_count())
    part_func = partial(func, *args, **kwargs)

    with Pool(os.cpu_count()) as pool:
        data = list(pool.imap(part_func, x))

    return data

def write_requirements():
    os.system("pip freeze > reqs_all.txt")
    reqs_exact = list()
    with open('reqs_all.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            reqs_exact.append(line.strip())
    reqs = ['matplotlib',
            'numpy',
            'pandas',
            'pymatgen',
            'scikit-learn',
            'mastml']
    with open('requirements.txt', 'w') as f:
        for req in reqs:
            for req_exact in reqs_exact:
                if req == req_exact.split('==')[0]:
                    f.write(req+'\n')
    return
