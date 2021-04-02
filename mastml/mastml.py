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
        else:
            self._update_metadata()
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
                self.savepath = f"{self.savepath}_{now.month:02d}_{now.day:02d}" \
                         f"_{now.hour:02d}_{now.minute:02d}_{now.second:02d}"
        os.makedirs(self.savepath)
        return

    def _initialize_metadata(self):
        self.mastml_metadata = OrderedDict()
        self.mastml_metadata['savepath'] = self.savepath
        return

    def _update_metadata(self):
        # Update with new entry: (1) module, (2) class, (3) path executed, (4) paths to data used ???
        pass

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
