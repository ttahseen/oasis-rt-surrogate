"""
This file loads raw simulation output data and returns the data in a
format ready for model training.
"""

import os
import h5py
import torch
import numpy as np
from tqdm import tqdm
from torch import Tensor

from vars_lw import *


########################### DATA PROCESSING ###########################


class DataProcesser:
    def __init__(self, datapath, data_vars, input_vars, target_vars, aux_vars=None):
        self.datapath = datapath
        self.data_vars = data_vars
        self.aux_vars = aux_vars  # Auxiliary vars which are not of same dimension as input vars
        self.input_vars = input_vars  # Variables which will be main inputs to model
        self.target_vars = target_vars  # Variables which will be outputs of model
        self.vars_used = None
        self.aux_vars_used = None

    @staticmethod
    def log_scaling(x: np.ndarray) -> np.ndarray:
        return np.log(x)

    @staticmethod
    def power_law_scaling(x: np.ndarray, k: float) -> np.ndarray:
        return x**k

    @staticmethod
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

    def load_data(self, path: str) -> dict:
        """
        This function loads and returns the raw simulation output data as
        a list of dictionaries containing data across different timesteps.

        Args:
            path (str): location of data
        """

        # Checks all files in output folder
        files = [f for f in os.listdir(path) if "oasis_output_Venus_" in f]

        outputs = {}
        # Loads all output files and saves them within some object
        for f in tqdm(files): 
            timestep = int(f.split("Venus_")[1].split(".h5")[0])
            output = DataProcesser.extract_data(os.path.join(path, f))
            output["timestep"] = timestep
            # outputs.append(output)
            outputs[timestep] = output

        grid = DataProcesser.extract_data(os.path.join(path, "oasis_output_grid_Venus.h5"))

        return outputs, grid

    def format_data(self, data: list[dict], grid, resample_data=True) -> tuple[Tensor, Tensor, Tensor]:
        """
        This function takes input of raw simulation output data and
        reformats this data in terms of pairs of values for all physical
        variables for each atmospheric column corresponding to values at
        time=t and time=t+1. The data is returned as a Tensor.

        resample_data:  resample evenly across range of cosz (raw data has
                        50% of data with cosz=0 and rest as evenly distributed
                        across 0<cosz<1)

        Returns:
            formatted_data: Tensor, shape=(nt, B, T, d)
                nt: number of timesteps
                B: batch size == number of columns
                T: sequence size == number of boundaries
                d: feature size == number of physical variables
            auxiliary_data: Tensor, shape=(nt, B, s)
                s: feature size == number of surface variables
        """

        print("==Variables==")
        for v in self.input_vars:
            print(v)

        input_vars = self.input_vars
        aux_vars = self.aux_vars
        target_vars = self.target_vars

        nt = len(data)  # Number of timesteps
        nlev = 50  # NOTE: Number of atmospheric levels *boundaries*
        nlay = nlev - 1
        ncol = 10242  # Number of atmospheric columns
        nvars = len(input_vars)  # Number of physical variables per atmospheric cell
        nauxvars = len(aux_vars)
        ntargets = len(target_vars)

        # Handling data from non-consecutive timesteps
        timesteps = sorted(data.keys())

        gap_start = []
        npairs = 0  # Number of pairs of consecutive timesteps
        for ix in range(1, nt):
            if not (timesteps[ix] - timesteps[ix - 1]) == 1:
                gap_start.append(ix)
            else:
                npairs += 1

        # Initialise numpy array
        input_data = np.zeros(shape=(nt, ncol, nlay, nvars))
        auxiliary_data = np.empty(shape=(nt, ncol, nauxvars))
        target_data = np.empty(shape=(nt, ncol, nlev, ntargets))
        altitudeh_data = np.empty(shape=(nt, ncol, nlev))

        # Populate numpy array with data
        for t in range(len(timesteps)):

            dt = data[timesteps[t]]

            for av, avar in enumerate(aux_vars):
                auxiliary_data[t, :, av] = dt[avar]

            for v, var in enumerate(input_vars):
                if len(dt[var]) == (ncol * nlay):
                    input_data[t, :, :, v] = np.reshape(dt[var], newshape=(ncol, nlay))
                if len(dt[var]) == (ncol * nlev):
                    input_data[t, :, :, v] = np.reshape(dt[var], newshape=(ncol, nlev))[:, :-1]

            for tv, tvar in enumerate(target_vars):
                target_data[t, :, :, tv] = np.reshape(dt[tvar], newshape=(ncol, nlev))

        # Rescale inputs
        for v, var in enumerate(input_vars):
            if var in ["Temperature", "Pressure"]:
                input_data[:, :, :, v] = DataProcesser.log_scaling(input_data[:, :, :, v])
            elif var == "Rho":
                input_data[:, :, :, v] = DataProcesser.power_law_scaling(input_data[:, :, :, v], 0.25)
            elif "flux" in var:
                var_scaling = input_data[:, :, -1, v, np.newaxis]
                input_data[:, :, :, v] = input_data[:, :, :, v] / var_scaling
    

        for tv, tvar in enumerate(target_vars):
            tvar_scaling = target_data[:, :, 0, tv, np.newaxis]
            target_data[:, :, :, tv] = target_data[:, :, :, tv] / tvar_scaling
            target_data = np.nan_to_num(
                target_data
            )  # TODO: WRITE THIS IN WAY TO AVOID NANS RATHER THAN JUST FILLING NANS AFTER
            print(target_data.shape)
            print(np.sum(target_data == 0))

        # Add altitude as vector input

        altitudeh_data = np.expand_dims(grid["Altitudeh"], axis=0)
        altitudeh_data = np.repeat(altitudeh_data, repeats=ncol, axis=0)
        altitudeh_data = np.expand_dims(altitudeh_data, axis=0)
        altitudeh_data = np.repeat(altitudeh_data, repeats=nt, axis=0)

        altitude_data = np.expand_dims(grid["Altitude"], axis=0)
        print("Unscaled altitudes: ", altitude_data)
        altitude_data = altitude_data / altitude_data.max()
        print("Rescaled altitudes: ", altitude_data)
        altitude_data = np.repeat(altitude_data, repeats=ncol, axis=0)
        altitude_data = np.expand_dims(altitude_data, axis=0)
        altitude_data = np.repeat(altitude_data, repeats=nt, axis=0)
        
        input_data = np.delete(arr=input_data, obj=gap_start, axis=0)
        auxiliary_data = np.delete(arr=auxiliary_data, obj=gap_start, axis=0)
        target_data = np.delete(arr=target_data, obj=gap_start, axis=0)
        altitudeh_data = np.delete(arr=altitudeh_data, obj=gap_start, axis=0)
        altitude_data = np.delete(arr=altitude_data, obj=gap_start, axis=0)

        # Add altitude to input
        input_data = np.concatenate((input_data, altitude_data[:, :, :, None]), axis=3)

        print("==========INPUT_DATA===========", input_data.shape)
       	print("==========ALTITUDE_DATA===========", altitude_data.shape)
        input_data, auxiliary_data, target_data, altitudeh_data = [
            torch.from_numpy(dataset).type(torch.float)
            for dataset in (input_data, auxiliary_data, target_data, altitudeh_data)
        ]

        self.inputs = torch.flatten(input_data[:, :, :, :], start_dim=0, end_dim=1)
        self.aux_inputs = torch.flatten(auxiliary_data[:, :, :], start_dim=0, end_dim=1)
        self.targets = torch.flatten(target_data[:, :, :, :], start_dim=0, end_dim=1)
        self.altitudeh = torch.flatten(altitudeh_data[:, :, :], start_dim=0, end_dim=1)

    def test_train_split(self, savepath, training_frac: float = 0.7, val_frac: float = 0.15):
        """
        Args:
            data: Tensor, shape=(nt, B, T, d)
                nt: number of timesteps
                B: batch size == number of columns
                T: sequence size == number of boundaries
                d: feature size == number of physical variables
            aux_data: Tensor, shape=(nt, B, s)
                s: feature size == number of surface variables
            training_frac: float in range [0,1]
                Fraction of data to use for train set. Test set will be of fraction 1-training_frac.
        """

        ns, T, d = self.inputs.shape
        num_train_examples = int(ns * training_frac)
        num_val_examples = int(ns * val_frac)

        inputs = self.inputs
        aux_inputs = self.aux_inputs
        targets = self.targets
        altitudeh = self.altitudeh

        # Randomly shuffle batches
        torch.manual_seed(42)
        shuffle_ix = torch.randperm(inputs.size()[0])
        inputs = inputs[shuffle_ix, :, :]
        targets = targets[shuffle_ix, :, :]
        altitudeh = altitudeh[shuffle_ix, :]

        train_x, train_y, train_altitudeh = (
            inputs[:num_train_examples, :, :],
            targets[:num_train_examples, :, :],
            altitudeh[:num_train_examples, :],
        )

        val_x, val_y, val_altitudeh = (
            inputs[num_train_examples : num_train_examples + num_val_examples, :, :],
            targets[num_train_examples : num_train_examples + num_val_examples, :, :],
            altitudeh[num_train_examples : num_train_examples + num_val_examples, :],
        )

        test_x, test_y, test_altitudeh = (
            inputs[num_train_examples + num_val_examples :, :, :],
            targets[num_train_examples + num_val_examples :, :, :],
            altitudeh[num_train_examples + num_val_examples :, :],
        )

        with open(os.path.join(savepath, "train_x.pt"), "wb") as f:
            torch.save(train_x, f)

        with open(os.path.join(savepath, "test_x.pt"), "wb") as f:
            torch.save(test_x, f)

        with open(os.path.join(savepath, "val_x.pt"), "wb") as f:
            torch.save(val_x, f)

        with open(os.path.join(savepath, "train_y.pt"), "wb") as f:
            torch.save(train_y, f)

        with open(os.path.join(savepath, "test_y.pt"), "wb") as f:
            torch.save(test_y, f)

        with open(os.path.join(savepath, "val_y.pt"), "wb") as f:
            torch.save(val_y, f)

        if type(aux_inputs) == Tensor:
            aux_inputs[shuffle_ix, :]
            train_aux_x = aux_inputs[:num_train_examples, :]
            val_aux_x = aux_inputs[num_train_examples : num_train_examples + num_val_examples, :]
            test_aux_x = aux_inputs[num_train_examples + num_val_examples :, :]

            with open(os.path.join(savepath, "train_aux_x.pt"), "wb") as f:
                torch.save(train_aux_x, f)

            with open(os.path.join(savepath, "test_aux_x.pt"), "wb") as f:
                torch.save(test_aux_x, f)

            with open(os.path.join(savepath, "val_aux_x.pt"), "wb") as f:
                torch.save(val_aux_x, f)

        with open(os.path.join(savepath, "train_altitudeh.pt"), "wb") as f:
            torch.save(train_altitudeh, f)

        with open(os.path.join(savepath, "test_altitudeh.pt"), "wb") as f:
            torch.save(test_altitudeh, f)

        with open(os.path.join(savepath, "val_altitudeh.pt"), "wb") as f:
            torch.save(val_altitudeh, f)

    def process(self):
        print("=============LOADING DATA=============")

        data, grid = self.load_data(self.datapath)

        print("============PROCESSING DATA===========")
        self.format_data(data, grid)

        formatted_data_path = os.path.join(self.datapath, "rnn_lw")
        if not os.path.exists(formatted_data_path):
            os.makedirs(formatted_data_path)

        print("=============SAVING DATA=============")
        self.test_train_split(
            training_frac=0.7,
            val_frac=0.15,
            savepath=formatted_data_path,
        )
        print("Savepath: ", formatted_data_path)

if __name__ == "__main__":
    processer = DataProcesser(
        datapath=DATAPATH,
        input_vars=INPUT_VARS,
        target_vars=TARGET_VARS,
        aux_vars=AUX_VARS,
    )

    processer.process()
