"""
This file contains useful functions for data manipulation and physics
calculations, utilised in model training and evaluation.
"""

import os
import torch
import tensorflow as tf
import tensorflow.keras.backend as K

############################# DATA LOADERS ############################


# def load_input_data(datapath: str, dataset: str = "train") -> tuple[torch.Tensor, torch.Tensor]:
def load_input_data(datapath, dataset = "train"):
    
    if dataset == "train":
        with open(os.path.join(datapath, "train_x.pt"), "rb") as f:
            train_x = torch.load(f)

        with open(os.path.join(datapath, "val_x.pt"), "rb") as f:
            val_x = torch.load(f)

        with open(os.path.join(datapath, "train_y.pt"), "rb") as f:
            train_y = torch.load(f)

        with open(os.path.join(datapath, "val_y.pt"), "rb") as f:
            val_y = torch.load(f)

        return train_x, train_y, val_x, val_y

    if dataset == "test":
        with open(os.path.join(datapath, "test_x.pt"), "rb") as f:
            test_x = torch.load(f)

        with open(os.path.join(datapath, "test_y.pt"), "rb") as f:
            test_y = torch.load(f)

        return test_x, test_y



# def load_aux_data(datapath: str, dataset: str = "train") -> tuple[torch.Tensor, torch.Tensor]:
def load_aux_data(datapath, dataset = "train"):
    if dataset == "train":
        with open(os.path.join(datapath, "train_aux_x.pt"), "rb") as f:
            train_aux_x = torch.load(f)

        with open(os.path.join(datapath, "val_aux_x.pt"), "rb") as f:
            val_aux_x = torch.load(f)

        return train_aux_x, val_aux_x

    if dataset == "test":
        with open(os.path.join(datapath, "test_aux_x.pt"), "rb") as f:
            test_aux_x = torch.load(f)

        return test_aux_x

# def convert_pt_to_tf_tensor(pt_tensor: torch.Tensor):
def convert_pt_to_tf_tensor(pt_tensor):
    np_tensor = pt_tensor.numpy()
    tf_tensor = tf.convert_to_tensor(np_tensor)

    return tf_tensor

def convert_tf_to_pt_tensor(tf_tensor):
    np_tensor = tf_tensor.numpy()
    pt_tensor = torch.from_numpy(np_tensor)

    return pt_tensor


######################### METRICS AND LOSSES #########################


def rmse_flux(y_true, y_pred):

    err_flux = K.mean(K.abs(y_true - y_pred), axis=1)

    norm0 = K.mean(K.abs(y_true[:, :, 0]), axis=1)
    norm1 = K.mean(K.abs(y_true[:, :, 1]), axis=1)

    err_flux1 = err_flux[:, 0] / norm0
    err_flux2 = err_flux[:, 1] / norm1
    err_flux = err_flux1 + err_flux2
    err_flux = tf.math.multiply(0.5, err_flux)

    return err_flux

def norm0(y_true, y_pred):
    norm0 = K.sqrt(K.mean(K.square(y_true[:, 0])))
    return norm0

def norm1(y_true, y_pred):
    norm1 = K.sqrt(K.mean(K.square(y_true[:, 1])))
    return norm1

def err_flux1(y_true, y_pred):
    err_flux1 = K.sqrt(K.mean(K.square(y_true[:,0]  - y_pred[:,0] ), axis=0))
    return err_flux1

def err_flux2(y_true, y_pred):
    err_flux2 = K.sqrt(K.mean(K.square(y_true[:,1] - y_pred[:,1]), axis=0))
    return err_flux2

def rmse_derivative(y_true, y_pred):

    dif_true = y_true[:, :-1, :] - y_true[:, 1:, :]
    dif_pred = y_pred[:, :-1, :] - y_pred[:, 1:, :]
    err_dif = K.mean(K.abs(dif_true - dif_pred), axis=1)

    norm0 = K.mean(K.abs(dif_true[:, :, 0]), axis=1)
    norm1 = K.mean(K.abs(dif_true[:, :, 1]), axis=1)

    err_dif1 = err_dif[:, 0] / norm0
    err_dif2 = err_dif[:, 1] / norm1

    err_dif = err_dif1 + err_dif2
    err_dif = tf.math.multiply(0.5, err_dif)

    return err_dif

def norm0_dif(y_true, y_pred):
    dif_true = y_true[:, :-1, :] - y_true[:, 1:, :]
    norm0 = K.mean(K.abs(dif_true[:, :, 0]), axis=1)
    return norm0

def norm1_dif(y_true, y_pred):
    dif_true = y_true[:, :-1, :] - y_true[:, 1:, :]
    norm1 = K.mean(K.abs(dif_true[:, :, 1]), axis=1)
    return norm1

def flux_up(y_true, y_pred):
    return K.mean(y_true[:, :, 0])

def flux_down(y_true, y_pred):
    return K.mean(y_true[:, :, 1])

def custom_loss(y_true, y_pred, alpha=0.5):
    err_flux = rmse_flux(y_true, y_pred)
    if alpha == 1:
        err_dif = 0
    else:
        err_dif = rmse_derivative(y_true, y_pred)

    return alpha * err_flux + (1 - alpha) * err_dif
