"""
This file takes raw data across timesteps of the original simulation,
computes gas opacity parameters across all grid elements for all columns,
and preprocesses this data ready for model training.
"""

import os
import h5py
import torch
import numpy as np
from tqdm import tqdm
from torch import Tensor

from .helper_scripts_optical.vars import *
from .helper_scripts_optical.model_const import *
from .helper_scripts_optical.get_opacities import generate_opacities_arrays

########################### DATA PROCESSING ###########################


def numpy_to_torch(input):
    if type(input) == np.ndarray:
        return Tensor(input)

    elif type(input) == tuple:
        return (Tensor(input[0]), Tensor(input[1]))

    elif type(input) == list:
        if type(input[0]) == np.ndarray:
            return [Tensor(i) for i in input]

        elif type(input[0]) == tuple:
            return [(Tensor(i[0]), Tensor(i[1])) for i in input]


class DataProcesser:
    def __init__(
        self,
        # input_vars,
        # aux_vars,
        target_vars=TARGET_VARS,
        ncol=NCOL,
        nlev=NLEV,
        nlay=NLAY,
        nt=NT,
        datapath=DATAPATH,
        input_path=INPUT_PATH,
        savepath=os.path.join(DATAPATH, "rnn_sw", "optical"),
    ):
        self.input_path = input_path
        self.datapath = datapath
        self.savepath = savepath
        # self.input_vars = input_vars
        # self.aux_vars = aux_vars
        self.target_vars = target_vars
        self.ncol = ncol
        self.nlev = nlev
        self.nlay = nlay
        self.nt = nt

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

    def load_output_data(self, path: str = DATAPATH) -> dict:
        """
        This function loads and returns the raw simulation output data as
        a list of dictionaries containing data across different timesteps.

        Args:
            path (str): location of data
        """

        # Checks all files in data folder
        files = [f for f in os.listdir(path) if "oasis_output_Venus_" in f]

        outputs = {}
        # Loads all output files and saves them within some object
        for f in files:
            timestep = int(f.split("Venus_")[1].split(".h5")[0])
            output = DataProcesser.extract_data(os.path.join(path, f))
            output["timestep"] = timestep
            # outputs.append(output)
            outputs[timestep] = output

        grid = DataProcesser.extract_data(os.path.join(path, "oasis_output_grid_Venus.h5"))
        return outputs, grid

    def load_input_data(self, path: str = INPUT_PATH) -> dict:
        """
        This function loads and returns the raw simulation input data as
        a list of dictionaries.

        Args:
            path (str): location of data
        """

        files = os.listdir(path)

        for f in ["oasis_planet_parameters_Venus.h5", "opacities_venus.h5", "stars.h5"]:
            if f not in files:
                raise FileNotFoundError(f"{f} not found in directory.")

        planet = DataProcesser.extract_data(os.path.join(path, "oasis_planet_parameters_Venus.h5"))
        opacities = DataProcesser.extract_data(os.path.join(path, "opacities_venus.h5"))
        stars = DataProcesser.extract_data(os.path.join(path, "stars.h5"))

        return planet, opacities, stars

    def get_targets(self, outputs):
        nt = self.nt
        ncol = self.ncol
        nlev = self.nlev

        # Targets
        targets = np.zeros(shape=(nt, ncol, nlev, len(self.target_vars)))

        for t in tqdm(range(1, nt)):
            for v, var in enumerate(self.target_vars):
                targets[t, :, :, v] = np.reshape(outputs[t][var], newshape=(ncol, nlev))

        return targets

    def get_opacities_for_raw_data(
        self,
        opacities_vector_vars=["opac", "opac_ext", "opac_ray"],
        opacities_scalar_vars=["cosz_d", "alb_surf_sw"],
    ):
        """
        This function computes and saves opacities arrays across all timesteps available, and then
        reloads this data into single Tensor.
        """
        opacities_path = os.path.join(self.datapath, "rnn_sw", "optical", "optical_variables")

        if not os.path.isdir(opacities_path):
            print("     Generating opacities....", flush=True)
            # Compute and save opacities arrays across timesteps, if opacities_dir is empty
            generate_opacities_arrays(datapath=self.datapath, input_path=self.input_path)

        nt = self.nt
        nvar = len(opacities_vector_vars)
        ncol = self.ncol
        nlev = self.nlev
        nlay = self.nlay

        # Create Tensor in which to store opacities data across timesteps
        opacities = np.zeros(shape=(nt, ncol, nlev, nvar))
        cosz = np.zeros(shape=(nt, ncol))
        alb_surf_sw = np.zeros(shape=(nt, ncol))

        # Load data into Tensor
        for t in tqdm(range(1, nt)):
            opac = np.load(os.path.join(opacities_path, f"opac-{t}.npy"))
            opac_ray = np.load(os.path.join(opacities_path, f"opac_ray-{t}.npy"))
            opac_ext = np.load(os.path.join(opacities_path, f"opac_ext-{t}.npy"))
            cosz_d = np.load(os.path.join(opacities_path, f"cosz_d-{t}.npy"))
            alb_surf_sw_3d = np.load(os.path.join(opacities_path, f"alb_surf_sw_3D-{t}.npy"))

            opacities[t, :, :, 0] = opac
            opacities[t, :, :, 1] = opac_ray
            opacities[t, :, :, 2] = opac_ext

            cosz[t, :] = cosz_d
            alb_surf_sw[t, :] = alb_surf_sw_3d

        return opacities, cosz, alb_surf_sw

    def scale_targets(self, targets):
        scaled_targets = np.zeros(np.shape(targets))

        # Scale downwelling flux by top level
        scaled_targets[:, :, 0] = targets[:, :, 0] / np.repeat(
            targets[:, -1, 0, np.newaxis], repeats=targets.shape[1], axis=1
        )

        # Scale upwelling flux by top level
        scaled_targets[:, :, 1] = targets[:, :, 1] / np.repeat(
            targets[:, -1, 1, np.newaxis], repeats=targets.shape[1], axis=1
        )
        
        return scaled_targets

    def scale_opacities(self, opacities, cosz, alb_surf_sw):
        scalings = []
        for var in range(2):
            min_val = np.min(opacities[:, :, var])
            opacities[:, :, var] = opacities[:, :, var] / min_val
            scalings.append(min_val)

        max_val = np.max(opacities[:, :, 2])
        opacities[:, :, 2] = opacities[:, :, 2] / max_val
        scalings.append(max_val)

        self.scalings = scalings
        # cosz doesn't need to be scaled

        # alb_surf_sw shouldn't need to be scaled? is it between [0, 1]?

        return opacities, cosz, alb_surf_sw

    def merge_time_and_column_axes(self, opacities, cosz, alb_surf_sw, targets):
        nt = self.nt
        ncol = self.ncol
        nlev = self.nlev
        nsamples = nt * ncol
        nvars = opacities.shape[-1]
        noutput = targets.shape[-1]

        # Reshape inputs and targets to have dim1 == nt*ncol
        opacities = np.reshape(opacities, (nsamples, nlev, nvars))  # Check if this is correct
        cosz = np.reshape(cosz, (nsamples))  # Check if this is correct
        alb_surf_sw = np.reshape(alb_surf_sw, (nsamples))  # Check if this is correct
        targets = np.reshape(targets, (nsamples, nlev, noutput))

        return opacities, cosz, alb_surf_sw, targets

    def remove_zero_cosz_samples(self, opacities, cosz, alb_surf_sw, targets):
        nonzero_cosz_indices = cosz != 0

        return (
            opacities[nonzero_cosz_indices, :, :],
            cosz[nonzero_cosz_indices],
            alb_surf_sw[nonzero_cosz_indices],
            targets[nonzero_cosz_indices, :, :],
        )

    def test_train_val_split(self, opacities, cosz, alb_surf_sw, targets, train_frac=0.7, val_frac=0.15):
        nsamples = opacities.shape[0]

        num_train_samples = int(train_frac * nsamples)
        num_val_samples = int(val_frac * nsamples)
        num_test_samples = nsamples - num_train_samples - num_val_samples

        # # Reshape inputs and targets to have dim1 == nt*ncol
        # opacities = np.reshape(opacities, (nsamples, nlev, nvars))  # Check if this is correct
        # cosz = np.reshape(cosz, (nsamples))  # Check if this is correct
        # alb_surf_sw = np.reshape(alb_surf_sw, (nsamples))  # Check if this is correct
        # targets = np.reshape(targets, (nsamples, -1))

        # Shuffle
        indices = np.arange(nsamples)
        np.random.shuffle(indices)

        opacities = opacities[indices, :, :]
        cosz = cosz[indices]
        alb_surf_sw = alb_surf_sw[indices]
        targets = targets[indices, :, :]

        inputs = np.zeros(shape=(opacities.shape[0], opacities.shape[1], opacities.shape[2]+1))
        inputs[:, :, :-1] = opacities[:, :, :]
        inputs[:, :, -1] = np.repeat(cosz[:, np.newaxis], opacities.shape[1], 1)

        aux_inputs = np.zeros(shape=(nsamples, 2))
        aux_inputs[:, 0] = cosz[:]
        aux_inputs[:, 1] = alb_surf_sw[:]

        # Split
        train_y = targets[:num_train_samples, :, :]
        val_y = targets[num_train_samples : num_train_samples + num_val_samples, :, :]
        test_y = targets[num_train_samples + num_val_samples :, :, :]

        train_aux_x = aux_inputs[:num_train_samples, :]
        val_aux_x = aux_inputs[num_train_samples : num_train_samples + num_val_samples, :]
        test_aux_x = aux_inputs[num_train_samples + num_val_samples :, :]

        train_x = inputs[:num_train_samples, :, :]
        val_x = inputs[num_train_samples : num_train_samples + num_val_samples, :, :]
        test_x = inputs[num_train_samples + num_val_samples :, :, :]

        return train_x, val_x, test_x, train_aux_x, val_aux_x, test_aux_x, train_y, val_y, test_y

    def process(self):
        """Main function."""
	
        print("Loading opacities....", flush=True)
        opacities, cosz, alb_surf_sw = self.get_opacities_for_raw_data()
        print("==========OPACITIES SHAPE==========", opacities.shape)
        print("Loading output data....", flush=True)
        outputs, grid = self.load_output_data()
        targets = self.get_targets(outputs)

        opacities, cosz, alb_surf_sw, targets = self.merge_time_and_column_axes(
            opacities, cosz, alb_surf_sw, targets
        )
       	print("==========OPACITIES SHAPE==========", opacities.shape)
        # Remove samples with cosz==0
        print("Removing samples with no incident solar flux....", flush=True)
        opacities, cosz, alb_surf_sw, targets = self.remove_zero_cosz_samples(
            opacities, cosz, alb_surf_sw, targets
        )

        print("Scaling inputs and targets....", flush=True)
        opacities, cosz, alb_surf_sw = self.scale_opacities(opacities, cosz, alb_surf_sw)
        targets = self.scale_targets(targets)
       	print("==========OPACITIES SHAPE==========", opacities.shape)
        print("Splitting data into test, train, val....", flush=True)
        (
            train_x,
            val_x,
            test_x,
            train_aux_x,
            val_aux_x,
            test_aux_x,
            train_y,
            val_y,
            test_y,
        ) = self.test_train_val_split(opacities, cosz, alb_surf_sw, targets)
        train_x, val_x, test_x, train_aux_x, val_aux_x, test_aux_x, train_y, val_y, test_y = numpy_to_torch(
            [train_x, val_x, test_x, train_aux_x, val_aux_x, test_aux_x, train_y, val_y, test_y]
        )

        # Save datasets
        print("Saving datasets....", flush=True)
        datasets = (
            train_x,
            val_x,
            test_x,
            train_aux_x,
            val_aux_x,
            test_aux_x,
            train_y,
            val_y,
            test_y,
        )

        dataset_labels = (
            "train_x",
            "val_x",
            "test_x",
            "train_aux_x",
            "val_aux_x",
            "test_aux_x",
            "train_y",
            "val_y",
            "test_y",
        )

        if not os.path.isdir(self.savepath):
            os.makedirs(self.savepath)

        for ix in range(len(datasets)):
            with open(os.path.join(self.savepath, f"{dataset_labels[ix]}.pt"), "wb") as f:
                torch.save(datasets[ix], f)

        # Write scalings to file
        with open(os.path.join(self.savepath, "opacities_scalings.txt"), "w") as f:
            for scaling in self.scalings:
                f.write(f"{scaling}\n")


if __name__ == "__main__":
    dp = DataProcesser()
    dp.process()
