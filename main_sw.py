"""
*** FUNCTIONS NEED TO BE FILLED OUT ***

This Python file ingests raw data, preprocesses it, loads the trained model, and then uses the model to make predictions.
Predictions are then post-processed, and returned.
"""

import os
import json
import numpy as np
import tensorflow as tf

################################################################################################################

def log_scaling(x: np.ndarray) -> np.ndarray:
    return np.log(x)

def power_law_scaling(x: np.ndarray, k: float) -> np.ndarray:
    return x**k

def standard_scaling(x: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    return (x - xmin) / (xmax - xmin)

################################################################################################################

def preprocess_data(pressure, temperature, rho, sTemperature, cosz, alb_surf_sw, rmin, rmax, pmin, pmax, tmin, tmax):
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
    nauxvars = 5

    # Check shape of inputs
    if not pressure.shape == temperature.shape == rho.shape:
        raise ValueError("Pressure, temperature, and density arrays must have the same shape.")
    if not sTemperature.shape == cosz.shape == alb_surf_sw.shape:
        raise ValueError("sTemperature, cosz, and alb_surf_sw arrays must have the same shape.")
    if not pressure.shape == (ncol, nlay):
        raise ValueError("Pressure, temperature, and density arrays must have shape (ncol, nlay).")
    if not sTemperature.shape == (ncol,):
        raise ValueError("sTemperature, cosz, and alb_surf_sw arrays must have shape (ncol,).")

    sPressure = pressure[0, :]
    sRho = rho[0, :]

    # Scale data
    rho[:, :] = power_law_scaling(rho[:, :], 0.25) / power_law_scaling(sRho[:, None], 0.25)
    temperature[:, :] = log_scaling(temperature[:, :]) / log_scaling(sTemperature[:, None])
    pressure[:, :] = log_scaling(pressure[:, :]) / log_scaling(sPressure[:, None])

    # Scale sTemperature, sPressure, sRho
    # TODO: Replace this standard scaling with physically motivated scaling (and retrain model)
    sRho = standard_scaling(sRho, rmin, rmax)
    sPressure = standard_scaling(sPressure, pmin, pmax)
    sTemperature = standard_scaling(sTemperature, tmin, tmax)

    inputs_main = np.zeros(shape=(ncol, nlay, nvars))
    inputs_aux = np.zeros(shape=(ncol, nauxvars))

    inputs_main[:, :, 0] = rho[:, :]
    inputs_main[:, :, 1] = temperature[:, :]
    inputs_main[:, :, 2] = pressure[:, :]

    inputs_aux[:, 0] = cosz[:]
    inputs_aux[:, 1] = alb_surf_sw[:]
    inputs_aux[:, 2] = sRho[:]
    inputs_aux[:, 3] = sPressure[:]
    inputs_aux[:, 4] = sTemperature[:]

    return inputs_main, inputs_aux

def load_model(model_path):
    """
    Loads the trained model from the specified path.

    Args:
    - model_path: path to the trained model

    Returns:
    - trained model
    """
    if not os.path.exists(model_path):
        raise ValueError("Model path does not exist.")

    return tf.keras.models.load_model(model_path) # TODO: Should model be saved in a different format?

def make_predictions(model, data):
    """
    Uses the trained model to make predictions on the data.

    Args:
    - model: trained model
    - data: preprocessed data

    Returns:
    - predictions
    """

    return model.predict(data)

def postprocess_predictions(predictions: np.ndarray, cosz: np.ndarray, m: float, c: float) -> np.ndarray:
    # return x * (m*cosz + c)
    return np.multiply(predictions, np.repeat(m*cosz[:, None] + c, 2, axis=1)) # TODO: Check that repeat has been done along correct axis

def main(pressure, temperature, rho, sTemperature, cosz, alb_surf_sw):
    # Load the model
    model_path = "/home/ucaptp0/oasis-rt-surrogate/analysis/trained-models/649968"
    model = load_model(model_path)

    # Load the data
    data = "data"
    preprocessed_data = preprocess_data(pressure, temperature, rho, sTemperature, cosz, alb_surf_sw)

    # Make predictions
    predictions = make_predictions(model, preprocessed_data)

    # Postprocess the predictions
    postprocessed_predictions = postprocess_predictions(predictions, cosz, m, c)

    return postprocessed_predictions

if __name__ == "__main__":
    main()
