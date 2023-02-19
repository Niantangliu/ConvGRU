"""Utilities related to disk I/O."""
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import sys
from collections import defaultdict


import os
import sys
import threading

from absl import logging

from keras.utils import keras_logging



try:
    import h5py
except ImportError:
    h5py = None


class HDF5Matrix(object):
    """Representation of HDF5 dataset to be used instead of a Numpy array.

    # Example

    ```python
        x_data = HDF5Matrix('input/file.hdf5', 'data')
        model.predict(x_data)
    ```

    Providing `start` and `end` allows use of a slice of the dataset.

    Optionally, a normalizer function (or lambda) can be given. This will
    be called on every slice of data retrieved.

    # Arguments
        datapath: string, path to a HDF5 file
        dataset: string, name of the HDF5 dataset in the file specified
            in datapath
        start: int, start of desired slice of the specified dataset
        end: int, end of desired slice of the specified dataset
        normalizer: function to be called on data when retrieved

    # Returns
        An array-like HDF5 dataset.
    """
    refs = defaultdict(int)

    def __init__(self, datapath, dataset, start=0, end=None, normalizer=None):
        if h5py is None:
            raise ImportError('The use of HDF5Matrix requires '
                              'HDF5 and h5py installed.')

        if datapath not in list(self.refs.keys()):
            f = h5py.File(datapath)
            self.refs[datapath] = f
        else:
            f = self.refs[datapath]
        self.data = f[dataset]
        self.start = start
        if end is None:
            self.end = self.data.shape[0]
        else:
            self.end = end
        self.normalizer = normalizer

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop = key.start, key.stop
            if start is None:
                start = 0
            if stop is None:
                stop = self.shape[0]
            if stop + self.start <= self.end:
                idx = slice(start + self.start, stop + self.start)
            else:
                raise IndexError
        elif isinstance(key, (int, np.integer)):
            if key + self.start < self.end:
                idx = key + self.start
            else:
                raise IndexError
        elif isinstance(key, np.ndarray):
            if np.max(key) + self.start < self.end:
                idx = (self.start + key).tolist()
            else:
                raise IndexError
        elif isinstance(key, list):
            if max(key) + self.start < self.end:
                idx = [x + self.start for x in key]
            else:
                raise IndexError
        else:
            raise IndexError
        if self.normalizer is not None:
            return self.normalizer(self.data[idx])
        else:
            return self.data[idx]

    @property
    def shape(self):
        """Gets a numpy-style shape tuple giving the dataset dimensions.

        # Returns
            A numpy-style shape tuple.
        """
        return (self.end - self.start,) + self.data.shape[1:]

    @property
    def dtype(self):
        """Gets the datatype of the dataset.

        # Returns
            A numpy dtype string.
        """
        return self.data.dtype

    @property
    def ndim(self):
        """Gets the number of dimensions (rank) of the dataset.

        # Returns
            An integer denoting the number of dimensions (rank) of the dataset.
        """
        return self.data.ndim

    @property
    def size(self):
        """Gets the total dataset size (number of elements).

        # Returns
            An integer denoting the number of elements in the dataset.
        """
        return np.prod(self.shape)


INTERACTIVE_LOGGING = threading.local()
INTERACTIVE_LOGGING.enable = keras_logging.INTERACTIVE_LOGGING_DEFAULT


def enable_interactive_logging():
    """Turn on interactive logging.
    When interactive logging is enabled, Keras displays logs via stdout.
    This provides the best experience when using Keras in an interactive
    environment such as a shell or a notebook.
    """
    INTERACTIVE_LOGGING.enable = True

def disable_interactive_logging():
    """Turn off interactive logging.
    When interactive logging is disabled, Keras sends logs to `absl.logging`.
    This is the best option when using Keras in a non-interactive
    way, such as running a training or inference job on a server.
    """
    INTERACTIVE_LOGGING.enable = False


def is_interactive_logging_enabled():
    """Check if interactive logging is enabled.
    To switch between writing logs to stdout and `absl.logging`, you may use
    `keras.utils.enable_interactive_logging()` and
    `keras.utils.disable_interactie_logging()`.
    Returns:
      Boolean (True if interactive logging is enabled and False otherwise).
    """
    # Use `getattr` in case `INTERACTIVE_LOGGING`
    # does not have the `enable` attribute.
    return getattr(
        INTERACTIVE_LOGGING, "enable", keras_logging.INTERACTIVE_LOGGING_DEFAULT
    )


def print_msg(message, line_break=True):
    """Print the message to absl logging or stdout."""
    if is_interactive_logging_enabled():
        if line_break:
            sys.stdout.write(message + "\n")
        else:
            sys.stdout.write(message)
        sys.stdout.flush()
    else:
        logging.info(message)

def ask_to_proceed_with_overwrite(filepath):
    """Produces a prompt asking about overwriting a file.

    # Arguments
        filepath: the path to the file to be overwritten.

    # Returns
        True if we can proceed with overwrite, False otherwise.
    """
    get_input = input
    if sys.version_info[:2] <= (2, 7):
        get_input = raw_input
    overwrite = get_input('[WARNING] %s already exists - overwrite? '
                          '[y/n]' % (filepath))
    while overwrite not in ['y', 'n']:
        overwrite = get_input('Enter "y" (overwrite) or "n" (cancel).')
    if overwrite == 'n':
        return False
    print('[TIP] Next time specify overwrite=True!')
    return True
