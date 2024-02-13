"""
This file takes input arguments of Job IDs corresponding to trained models, and finds the corresponding
checkpoint files, loads the test data, and evaluates the model on the test data. The predictions are then
saved to a file in a directory specified by the SAVEPATH variable.
"""

import argparse
import numpy as np
import os
import sys
import time
import torch
import datetime
import tensorflow as tf
import tensorflow_addons as tfa
from sys import platform
from tensorflow import keras

sys.path.append("/home/ucaptp0/oasis-rt-surrogate")

from utils import (
    get_timestamps_for_job_ids,
    get_checkpoint_files_for_timestamps,
    load_test_data,
    load_and_compile_model
)

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], "GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialised
        print(e)

####################################################################################################

SAVEPATH = os.path.join("/home/ucaptp0/oasis-rt-surrogate/analysis/trained-models/predictions")

####################################################################################################

# Parse input job_ids from command line
parser = argparse.ArgumentParser(description='Get model predictions')
parser.add_argument('job_ids', type=int, nargs='+', help='List of job IDs to get predictions for')
args = parser.parse_args()
job_ids = args.job_ids

# Get timestamps corresponding to input job_ids
timestamps_dict = get_timestamps_for_job_ids(job_ids)

# Using timestamps, find the corresponding checkpoint files
checkpoint_files, job_id_info = get_checkpoint_files_for_timestamps(timestamps_dict)

print(checkpoint_files)
print(job_id_info)

for jid in job_ids:
    schema = job_id_info[jid][0]
    inputs = job_id_info[jid][1]

    print("Job ID: {}".format(jid))
    print("Schema: {}".format(schema))
    print("Inputs: {}".format(inputs))

    # Load test data
    test_x, test_aux_x, test_y = load_test_data(schema=schema, inputs=inputs)

    # Load and compile the model
    model = load_and_compile_model(checkpoint_files[jid])

    # Evaluate the model on the test data
    loss = model.evaluate(
        x=[test_x, test_aux_x, test_y], y=test_y, verbose=1, batch_size=4096
    )
    print("Job {}: Loss on test data: {}".format(jid, loss))

    preds = model.predict(
        x=[test_x, test_aux_x, test_y], verbose=1, batch_size=4096
    )
        
    # Save the predictions
    with open(os.path.join(SAVEPATH, "predictions-{}.npy".format(jid)), "wb") as f:
        np.save(f, preds)
