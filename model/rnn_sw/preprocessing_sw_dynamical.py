"""
This file loads raw simulation output data and returns the data in a
format ready for model training.
"""

import os
import sys
import h5py
import json
import torch
import numpy as np
from tqdm import tqdm
from torch import Tensor

from model.rnn_sw.vars_sw import *

########################### DATA PROCESSING ###########################


class DataProcesser:
    def __init__(self, datapath, input_vars, target_vars, aux_vars=None):
        self.datapath = datapath
        self.aux_vars = aux_vars  # Auxiliary vars which are not of same dimension as input vars
        self.input_vars = input_vars  # Variables which will be main inputs to model
        self.target_vars = target_vars  # Variables which will be outputs of model
        self.custom_aux_vars = aux_vars
        self.default_aux_vars = ["scaled_sTemperature", "scaled_sPressure", "scaled_sRho"]
        self.all_aux_vars =  self.custom_aux_vars + self.default_aux_vars
        self.vars_used = None
        self.aux_vars_used = None

        # Load flux scale factor 
        flux_scale_factors = np.loadtxt(os.path.join(datapath, "constants", 'sw_flux_scaling_factors.txt'), unpack=True)
        self.flux_scale_factors = flux_scale_factors

    @staticmethod
    def log_scaling(x: np.ndarray) -> np.ndarray:
        return np.log(x)

    @staticmethod
    def power_law_scaling(x: np.ndarray, k: float) -> np.ndarray:
        return x**k
    
    @staticmethod
    def standard_scaling(x: np.ndarray) -> np.ndarray:
        vals = {
            "min": x.min(),
            "max": x.max(),
        }
        return (x - x.min()) / (x.max() - x.min()), vals

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
        target_vars = self.target_vars
        custom_aux_vars = self.custom_aux_vars
        default_aux_vars = self.default_aux_vars
        all_aux_vars = self.all_aux_vars

        nt = len(data)  # Number of timesteps
        nlev = 50  # NOTE: Number of atmospheric levels *boundaries*
        nlay = nlev - 1
        ncol = 10242  # Number of atmospheric columns
        nvars = len(input_vars)  # Number of physical variables per atmospheric cell
        nauxvars = len(all_aux_vars)
        ntargets = len(target_vars)

        print(f"Number of timesteps: {nt}", flush=True)
        print(f"Number of atmospheric levels: {nlev}", flush=True)
        print(f"Number of atmospheric layers: {nlay}", flush=True)
        print(f"Number of atmospheric columns: {ncol}", flush=True)
        print(f"Number of physical variables: {nvars}. These are: {input_vars}", flush=True)
        print(f"Number of surface variables: {nauxvars}. These are: {all_aux_vars}", flush=True)
        print(f"Number of target variables: {ntargets}. These are: {target_vars}", flush=True)

        # Handling data from non-consecutive timesteps
        timesteps = sorted(data.keys())

        gap_start = []
        npairs = 0  # Number of pairs of consecutive timesteps
        for ix in range(1, nt):
            if not (timesteps[ix] - timesteps[ix - 1]) == 1:
                gap_start.append(ix)
            else:
                npairs += 1
        # Check if there are non-consecutive timesteps
        if len(gap_start) > 0:
            raise ValueError("Non-consecutive timesteps found")

        # Initialise numpy array
        input_data = np.zeros(shape=(nt, ncol, nlay, nvars))
        auxiliary_data = np.empty(shape=(nt, ncol, nauxvars))
        target_data = np.empty(shape=(nt, ncol, nlev, ntargets))
        cosz = np.empty(shape=(nt, ncol))
        sTemperature = np.empty(shape=(nt, ncol))
        sPressure = np.empty(shape=(nt, ncol))
        sRho = np.empty(shape=(nt, ncol))
        simulation_time = np.empty(shape=(nt, ncol))
        lonlat = np.empty(shape=(nt, ncol, 2))

        # Populate numpy array with data
        for t in range(len(timesteps)):

            dt = data[timesteps[t]]

            for av, avar in enumerate(custom_aux_vars):
                auxiliary_data[t, :, av] = dt[avar]

            for v, var in enumerate(input_vars):
                if len(dt[var]) == (ncol * nlay):
                    input_data[t, :, :, v] = np.reshape(dt[var], newshape=(ncol, nlay))
                if len(dt[var]) == (ncol * nlev):
                    input_data[t, :, :, v] = np.reshape(dt[var], newshape=(ncol, nlev))[:, :-1]

            for tv, tvar in enumerate(target_vars):
                target_data[t, :, :, tv] = np.reshape(dt[tvar], newshape=(ncol, nlev))

            cosz[t, :] = np.reshape(dt["cosz"], newshape=(ncol,))
            sTemperature[t, :] = np.reshape(dt["sTemperature"], newshape=(ncol,))
            sPressure[t, :] = np.reshape(dt["Pressure"], newshape=(ncol, nlay))[:, 0]
            sRho[t, :] = np.reshape(dt["Rho"], newshape=(ncol, nlay))[:, 0]
            simulation_time[t, :] = np.full(ncol, dt["simulation_time"])
            lonlat[t, :, :] = np.reshape(grid['lonlat'], newshape=(ncol, 2))

        # Standard scaling of surface variables:
        scaled_sTemperature, vals_sTemperature = DataProcesser.standard_scaling(sTemperature)
        scaled_sRho, vals_sRho = DataProcesser.standard_scaling(sRho)
        scaled_sPressure, vals_sPressure = DataProcesser.standard_scaling(sPressure)

        # Populate rest of aux vars
        auxiliary_data[:, :, -1] = scaled_sTemperature
        auxiliary_data[:, :, -2] = scaled_sPressure
        auxiliary_data[:, :, -3] = scaled_sRho

        # Save scaling values in one
        scalings_dict = {"sTemperature": vals_sTemperature, "sPressure": vals_sPressure, "sRho": vals_sRho}
        # Save scalings_dict to file
        with open(os.path.join(self.datapath, "constants", "sw_surface_scaling_values.txt"), "w") as file:
            file.write(json.dumps(scalings_dict))

        if resample_data:
            num_non_zero_cols = np.count_nonzero(cosz[0, :])
            print("Initial number of columns with non-zero cosz: ", num_non_zero_cols, flush=True)
            for t in range(nt):
                num_non_zero_cols_t = np.count_nonzero(cosz[t, :])
                if num_non_zero_cols_t < num_non_zero_cols:
                    num_non_zero_cols = num_non_zero_cols_t

            print(f"Final number of columns with non-zero cosz: {num_non_zero_cols}", flush=True)

            # TODO: Replace this code with more robust sampling method
            new_input_data = np.zeros(shape=(nt, num_non_zero_cols, nlay, nvars))
            new_auxiliary_data = np.empty(shape=(nt, num_non_zero_cols, nauxvars))
            new_target_data = np.empty(shape=(nt, num_non_zero_cols, nlev, ntargets))
            new_sTemperature = np.empty(shape=(nt, num_non_zero_cols))
            new_sRho = np.empty(shape=(nt, num_non_zero_cols))
            new_sPressure = np.empty(shape=(nt, num_non_zero_cols))
            new_cosz = np.empty(shape=(nt, num_non_zero_cols))
            new_lonlat = np.empty(shape=(nt, num_non_zero_cols, 2))

            for t in range(nt):
                nonzero_indices = np.where(cosz[t, :] != 0)[0][:num_non_zero_cols]
                indices = nonzero_indices

                new_input_data[t, :, :, :] = input_data[t, indices, :, :]
                new_auxiliary_data[t, :, :] = auxiliary_data[t, indices, :]
                new_target_data[t, :, :, :] = target_data[t, indices, :, :]
                new_cosz[t, :] = cosz[t, indices]
                new_sTemperature[t, :] = sTemperature[t, indices]
                new_sPressure[t, :] = sPressure[t, indices]
                new_sRho[t, :] = sRho[t, indices]
                new_lonlat[t, :, :] = lonlat[t, indices, :]

            input_data = new_input_data
            auxiliary_data = new_auxiliary_data
            target_data = new_target_data
            cosz = new_cosz
            sTemperature = new_sTemperature
            sPressure = new_sPressure
            sRho = new_sRho
            lonlat = new_lonlat
            simulation_time = simulation_time[:, :num_non_zero_cols]

        # Rescale inputs
        for v, var in enumerate(input_vars):
            if var == "Temperature":
                input_data[:, :, :, v] = DataProcesser.log_scaling(input_data[:, :, :, v]) / DataProcesser.log_scaling(sTemperature[:, :, None])
            elif var == "Pressure":
                input_data[:, :, :, v] = DataProcesser.log_scaling(input_data[:, :, :, v]) / DataProcesser.log_scaling(sPressure[:, :, None])
            elif var == "Rho":
                input_data[:, :, :, v] = DataProcesser.power_law_scaling(input_data[:, :, :, v], 0.25) / DataProcesser.power_law_scaling(sRho[:, :, None], 0.25)

        # Rescale targets
        tvar_scaling = self.flux_scale_factors[0] * cosz[:, :, np.newaxis, np.newaxis] + self.flux_scale_factors[1]
        target_data[:, :, :, :] = target_data[:, :, :, :] / tvar_scaling
        target_data = np.nan_to_num(
            target_data
        )  # TODO: WRITE THIS IN WAY TO AVOID NANS RATHER THAN JUST FILLING NANS AFTER
        
        input_data = np.delete(arr=input_data, obj=gap_start, axis=0)
        auxiliary_data = np.delete(arr=auxiliary_data, obj=gap_start, axis=0)
        target_data = np.delete(arr=target_data, obj=gap_start, axis=0)

        input_data, auxiliary_data, target_data, cosz, lonlat, simulation_time = [
            torch.from_numpy(dataset).type(torch.float)
            for dataset in (input_data, auxiliary_data, target_data, cosz, lonlat, simulation_time)
        ]

        self.inputs = torch.flatten(input_data[:, :, :, :], start_dim=0, end_dim=1)
        self.aux_inputs = torch.flatten(auxiliary_data[:, :, :], start_dim=0, end_dim=1)
        self.targets = torch.flatten(target_data[:, :, :, :], start_dim=0, end_dim=1)
        self.cosz = torch.flatten(cosz[:, :], start_dim=0, end_dim=1)
        self.lonlat = torch.flatten(lonlat[:, :, :], start_dim=0, end_dim=1)
        self.simulation_time = torch.flatten(simulation_time[:, :], start_dim=0, end_dim=1)

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
        cosz = self.cosz
        lonlat = self.lonlat
        simulation_time = self.simulation_time

        # Randomly shuffle batches
        torch.manual_seed(42)
        shuffle_ix = torch.randperm(inputs.size()[0])
        inputs = inputs[shuffle_ix, :, :]
        targets = targets[shuffle_ix, :, :]
        cosz = cosz[shuffle_ix]
        lonlat = lonlat[shuffle_ix, :]
        simulation_time = simulation_time[shuffle_ix]

        train_x, train_y, train_cosz, train_lonlat, train_simulation_time = (
            inputs[:num_train_examples, :, :],
            targets[:num_train_examples, :, :],
            cosz[:num_train_examples],
            lonlat[:num_train_examples, :],
            simulation_time[:num_train_examples]
        )

        val_x, val_y, val_cosz, val_lonlat, val_simulation_time = (
            inputs[num_train_examples : num_train_examples + num_val_examples, :, :],
            targets[num_train_examples : num_train_examples + num_val_examples, :, :],
            cosz[num_train_examples : num_train_examples + num_val_examples],
            lonlat[num_train_examples : num_train_examples + num_val_examples, :],
            simulation_time[num_train_examples : num_train_examples + num_val_examples]
        )

        test_x, test_y, test_cosz, test_lonlat, test_simulation_time = (
            inputs[num_train_examples + num_val_examples :, :, :],
            targets[num_train_examples + num_val_examples :, :, :],
            cosz[num_train_examples + num_val_examples :],
            lonlat[num_train_examples + num_val_examples :, :],
            simulation_time[num_train_examples + num_val_examples :]
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

        with open(os.path.join(savepath, "train_cosz.pt"), "wb") as f:
            torch.save(train_cosz, f)

        with open(os.path.join(savepath, "test_cosz.pt"), "wb") as f:
            torch.save(test_cosz, f)

        with open(os.path.join(savepath, "val_cosz.pt"), "wb") as f:
            torch.save(val_cosz, f)

        with open(os.path.join(savepath, "train_lonlat.pt"), "wb") as f:
            torch.save(train_lonlat, f)
        
        with open(os.path.join(savepath, "test_lonlat.pt"), "wb") as f:
            torch.save(test_lonlat, f)

        with open(os.path.join(savepath, "val_lonlat.pt"), "wb") as f:
            torch.save(val_lonlat, f)

        with open(os.path.join(savepath, "train_simulation_time.pt"), "wb") as f:
            torch.save(train_simulation_time, f)

        with open(os.path.join(savepath, "test_simulation_time.pt"), "wb") as f:
            torch.save(test_simulation_time, f)

        with open(os.path.join(savepath, "val_simulation_time.pt"), "wb") as f:
            torch.save(val_simulation_time, f)

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

    def process(self):
        print("=============LOADING DATA=============")

        raw_data_path = os.path.join(self.datapath, "raw_data")
        data, grid = self.load_data(raw_data_path)

        print("============PROCESSING DATA===========")
        self.format_data(data, grid)

        formatted_data_path = os.path.join(self.datapath, "preprocessed_data", "rnn_sw", "dynamical")
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
        input_vars=INPUT_VARS_DYNAMICAL,
        target_vars=TARGET_VARS,
        aux_vars=AUX_VARS,
    )

    processer.process()
