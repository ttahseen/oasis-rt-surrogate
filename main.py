"""
This Python file ingests raw data, preprocesses it, loads the trained model, and then uses the model to make predictions.
Predictions are then post-processed, and returned.
"""

import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from sys import platform
from tensorflow.keras import optimizers

################################################################################################################

# Paths
SW_MODEL_PATH = "/home/ucaptp0/oasis-rt-surrogate/trained_models/rnn_sw/dynamical/649968"
LW_MODEL_PATH = "/home/ucaptp0/oasis-rt-surrogate/trained_models/rnn_lw/dynamical/649951"

# Constants
M_SW = 2.778805990703287989e+03
C_SW = 2.875793951916430527e-13

M_LW = 8.846453815075061300e+01
C_LW = -4.847616859839935205e+04

surf_scaling_factors = { # These scaling factors are defined during initial preprocessing stage prior to model training
    "sTemperature":
        {"min": 726.9722671994392, "max": 733.5182678798269},
    "sPressure":
        {"min": 9305885.197405094, "max": 9309114.726490144},
    "sRho":
        {"min": 67.54836313738393, "max": 68.11203268337228}}

RMIN, RMAX = surf_scaling_factors["sRho"]["min"], surf_scaling_factors["sRho"]["max"]
PMIN, PMAX = surf_scaling_factors["sPressure"]["min"], surf_scaling_factors["sPressure"]["max"]
TMIN, TMAX = surf_scaling_factors["sTemperature"]["min"], surf_scaling_factors["sTemperature"]["max"]

BASE_LR = 0.0001 
MAX_LR = 0.0005

################################################################################################################

def log_scaling(x: np.ndarray) -> np.ndarray:
    return np.log(x)

def power_law_scaling(x: np.ndarray, k: float) -> np.ndarray:
    return x**k

def standard_scaling(x: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    return (x - xmin) / (xmax - xmin)

################################################################################################################

def preprocess_data(
        pressure,
        temperature,
        rho,
        sTemperature,
        cosz,
        alb_surf_sw,
        alb_surf_lw,
        rmin=RMIN,
        rmax=RMAX,
        pmin=PMIN,
        pmax=PMAX,
        tmin=TMIN,
        tmax=TMAX
        ):
    """
    Preprocesses raw data for use in the model.

    Args:
    - pressure: shape (ncol, nlay)
    - temperature: shape (ncol, nlay)
    - rho: shape (ncol, nlay)
    - sTemperature: shape (ncol,)
    - cosz: shape (ncol,)
    - alb_surf_sw: shape (ncol,)

    Returns:
    - inputs_main: shape (ncol, nlay, nvars)
    - inputs_aux: shape (ncol, nauxvars)
    """
    ncol, nlay = 10242, 49
    nvars = 3
    sw_nauxvars = 5
    lw_nauxvars = 4

    # Check shape of inputs
    if not pressure.shape == temperature.shape == rho.shape:
        raise ValueError("Pressure, temperature, and density arrays must have the same shape.")
    if not sTemperature.shape == cosz.shape == alb_surf_sw.shape:
        raise ValueError("sTemperature, cosz, and alb_surf_sw arrays must have the same shape.")
    if not pressure.shape == (ncol, nlay):
        raise ValueError("Pressure, temperature, and density arrays must have shape (ncol, nlay).")
    if not sTemperature.shape == (ncol,):
        raise ValueError("sTemperature, cosz, and alb_surf_sw arrays must have shape (ncol,).")

    sPressure = np.copy(pressure[:, 0])
    sRho = np.copy(rho[:, 0])

    # Scale data
    rho[:, :] = power_law_scaling(rho[:, :], 0.25) / power_law_scaling(sRho[:, None], 0.25)
    temperature[:, :] = log_scaling(temperature[:, :]) / log_scaling(sTemperature[:, None])
    pressure[:, :] = log_scaling(pressure[:, :]) / log_scaling(sPressure[:, None])

    # Scale sTemperature, sPressure, sRho
    # TODO: Replace this standard scaling with physically motivated scaling (and retrain model)
    scaled_sRho = standard_scaling(sRho, rmin, rmax)
    scaled_sPressure = standard_scaling(sPressure, pmin, pmax)
    scaled_sTemperature = standard_scaling(sTemperature, tmin, tmax)

    inputs_main = np.zeros(shape=(ncol, nlay, nvars))
    sw_inputs_aux = np.zeros(shape=(ncol, sw_nauxvars))
    lw_inputs_aux = np.zeros(shape=(ncol, lw_nauxvars))

    inputs_main[:, :, 0] = rho[:, :]
    inputs_main[:, :, 1] = temperature[:, :]
    inputs_main[:, :, 2] = pressure[:, :]

    sw_inputs_aux[:, 0] = cosz[:]
    sw_inputs_aux[:, 1] = alb_surf_sw[:]
    sw_inputs_aux[:, 2] = scaled_sRho[:]
    sw_inputs_aux[:, 3] = scaled_sPressure[:]
    sw_inputs_aux[:, 4] = scaled_sTemperature[:]

    lw_inputs_aux[:, 0] = alb_surf_lw[:]
    lw_inputs_aux[:, 1] = scaled_sRho[:]
    lw_inputs_aux[:, 2] = scaled_sPressure[:]
    lw_inputs_aux[:, 3] = scaled_sTemperature[:]

    return inputs_main, sw_inputs_aux, lw_inputs_aux


def load_and_compile_model(model_path, base_lr=BASE_LR, max_lr=MAX_LR, steps_per_epoch=1000):
    """
    This function loads a model from a given path and compiles it.
    (Note: model has to be loaded and compiled in such a way as it uses custom metrics)
    """
    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")
    
    clr = tfa.optimizers.CyclicalLearningRate(
        initial_learning_rate=base_lr,
        maximal_learning_rate=max_lr,
        scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
        step_size=2 * steps_per_epoch,
    )

    if platform == "darwin":
        optim = optimizers.legacy.Adam(learning_rate=clr)
    else:
        optim = optimizers.Adam(learning_rate=clr)

    model = tf.keras.models.load_model(
        model_path,
        custom_objects=None,
        compile=False
        )
    
    model.compile(optimizer=optim)
    return model

def make_predictions(model, inputs_main, inputs_aux):
    """
    Uses the trained model to make predictions on the data.

    Args:
    - model: trained model
    - data: preprocessed data

    Returns:
    - predictions: shape (ncol, nlev, noutput)
        - noutput: 2, ["fnet_dn_sw_h", "fnet_up_sw_h"]
    """

    return model.predict(
        x=[inputs_main, inputs_aux], verbose=1, batch_size=4096
    )

def postprocess_predictions(predictions: np.ndarray, cosz: np.ndarray, m: float, c: float) -> np.ndarray:
    # return x * (m*cosz + c)
    # return np.multiply(predictions, np.repeat(m*cosz[:, None] + c, 2, axis=1)) # TODO: Check that repeat has been done along correct axis
    return np.multiply(predictions, (m*cosz[:, None, None] + c))

def main(pressure, temperature, rho, sTemperature, cosz, alb_surf_sw, alb_surf_lw):
    # Load the model
    sw_model_path = SW_MODEL_PATH
    lw_model_path = LW_MODEL_PATH
    sw_model = load_and_compile_model(sw_model_path)
    lw_model = load_and_compile_model(lw_model_path)

    # Preprocess input data
    inputs_main, sw_inputs_aux, lw_inputs_aux = preprocess_data(pressure, temperature, rho, sTemperature, cosz, alb_surf_sw, alb_surf_lw)

    # Make predictions
    sw_predictions = make_predictions(sw_model, inputs_main, sw_inputs_aux)
    lw_predictions = make_predictions(lw_model, inputs_main, lw_inputs_aux)

    # Postprocess the predictions
    sw_postprocessed_predictions = postprocess_predictions(sw_predictions, cosz, m=M_SW, c=C_SW)
    lw_postprocessed_predictions = postprocess_predictions(lw_predictions, sTemperature, m=M_LW, c=C_LW)

    return sw_postprocessed_predictions, lw_postprocessed_predictions

if __name__ == "__main__":
    main()
