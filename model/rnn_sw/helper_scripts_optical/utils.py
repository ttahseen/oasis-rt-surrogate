import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from wasabi import msg
from matplotlib import cm, colors
from matplotlib.animation import FuncAnimation

###########################################################################################


def extract_data(filepath: str) -> dict:
    """
    For a .h5 file specified by a filepath, this function returns
    data in the form of a dictionary.
    """

    results = {}
    with h5py.File(filepath, "r") as f:
        for key in f.keys():
            if type(f[key]) == h5py.Dataset:
                ds_arr = f[key][()]  # returns as a numpy array
                results[key] = ds_arr

            elif type(f[key]) == h5py.Group:
                group = f.get(key)
                gdict = {}
                for gkey in group.keys():
                    ds_arr = group[gkey][()]
                    gdict[gkey] = ds_arr
                results[key] = gdict

    return results
