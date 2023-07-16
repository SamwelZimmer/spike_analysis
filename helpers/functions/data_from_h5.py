import h5py
import matplotlib.pyplot as plt
from pprint import pprint
import pandas as pd

# the times which we want to look at (seconds)
slicing_start_timestamps = {
    "spontaneous": 500,
    "grp": 2000,
    "ttx": 4500
}

# the data is stored in hdf5 format
def print_structure(name, obj):
    print(name)

def get_data(sample_rate: int, file_path: str, data_path: str, duration: int=1000000, start_timestamp: int=0, show_structure: bool=False, as_dataframe: bool=True):

    with h5py.File(file_path, "r") as f:
        if show_structure:
            # view the hdf5 file structure
            f.visititems(print_structure)

        # get the channel data
        channel_data = f[data_path]

        # get a slice of the data
        data = channel_data[:, start_timestamp * sample_rate : (start_timestamp * sample_rate) + duration]

        if as_dataframe:
            return pd.DataFrame(data.T)

        return data