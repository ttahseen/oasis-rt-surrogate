
import os
import re
import sys
import time
import torch
import datetime
import tensorflow as tf
import tensorflow_addons as tfa
from sys import platform
from tensorflow import keras
from sys import platform
from tensorflow.keras import optimizers

sys.path.append("/home/ucaptp0/oasis-rt-surrogate")

from model.rnn_lw.vars_lw import BASE_LR, MAX_LR
from typing import List

####################################################################################################

# Define the directory where slurm-*.out files are stored
SLURM_OUTPUT_DIR = os.path.join("/home/ucaptp0/oasis-rt-surrogate/slurm-outputs")
CHECKPOINTS_DIR = os.path.join("/home/ucaptp0/oasis-rt-surrogate/checkpoints")
PREPROCESSED_DATAPATH = os.path.join("/home/ucaptp0/oasis-rt-surrogate/data/preprocessed_data")

####################################################################################################

def get_timestamps_for_job_ids(job_ids:list, directory=SLURM_OUTPUT_DIR) -> dict:
    """
    This function takes a list of job_ids corresponding to model training jobs.
    The function looks through corresponding slurm-*.out files and extracts the
    timestamp of the job.

    Inputs:
    - job_ids: A list of job_ids to get timestamps for
    - directory: The directory to look for slurm-*.out files

    Returns:
    - timestamp_dict: A dictionary with job_ids as keys and timestamps as values
    """
    # Iterate over each file
    timestamp_dict = {}
    for jid in job_ids:
        file = os.path.join(directory, "slurm-{}.out".format(jid))
        if not os.path.exists(file):
            print("File {} does not exist".format(file))
            continue
        with open(file, "r") as f:
            content = f.read()
            # Find the timestamp after "checkpoints-"
            match = re.search(r"checkpoints-(\d+).(\d+)", content)
            if match:
                timestamp = match.group(1) + "." + match.group(2)
                print("Timestamp in file {}: {}".format(jid, timestamp))
                timestamp_dict[jid] = timestamp

    return timestamp_dict

def get_checkpoint_files_for_timestamps(timestamps: dict, directory=CHECKPOINTS_DIR) -> List[dict]:
    """
    This function takes a dictionary of timestamps and finds the corresponding
    checkpoint files in the directory.

    Inputs:
    - timestamps: A dictionary with job_ids as keys and timestamps as values
    - directory: The directory to look for checkpoint files

    Returns:
    - checkpoint_files: A dictionary with job ids as keys and checkpoint files as values
    - job_id_info: A dictionary with job ids as keys and a list of schema (LW vs SW) and inputs
                   (dynamical vs optical) as values
    """
    # Iterate over each timestamp
    checkpoint_files = {}
    job_id_info = {}
    for jid, timestamp in timestamps.items():
        for schema in ["rnn_lw", "rnn_sw"]:
            for inputs in ["dynamical", "optical"]:
                # Find the checkpoint file
                checkpoint_subdir = os.path.join(directory, schema, inputs, "checkpoints-{}".format(timestamp))
                if os.path.exists(checkpoint_subdir):
                    print("Checkpoint file for timestamp {}: {}".format(timestamp, checkpoint_subdir))
                    epoch_dir = find_best_checkpoint(checkpoint_subdir)
                    checkpoint_files[jid] = epoch_dir
                    job_id_info[jid] = [schema, inputs]

    return checkpoint_files, job_id_info

def find_best_checkpoint(checkpoint_dir):
    """
    Given a directory containing checkpoints, this function finds the latest epoch 
    with a checkpoint file.

    Returns:
    - latest_epoch_dir: The path to the model file contained in the latest saved epoch
    """
    # Find the directory in checkpoint_dir corresponding to the latest epoch
    epoch_dirs = os.listdir(checkpoint_dir)
    epoch_dirs.sort()
    epoch_dirs.reverse()
    for epoch_dir in epoch_dirs:
        # Check if epoch_dir contains checkpoints.model.tf
        # if "checkpoint.model.tf" in os.listdir(os.path.join(checkpoint_dir, epoch_dir)):
        if "checkpoint.model.keras" in os.listdir(os.path.join(checkpoint_dir, epoch_dir)):
            # If yes, then this is the latest epoch
            latest_epoch_dir = epoch_dir
            print("Checkpoint found")
            return os.path.join(checkpoint_dir, latest_epoch_dir, "checkpoint.model.keras")
    raise FileNotFoundError("No checkpoint found")

def load_test_data(schema: str, inputs: str, preprocessed_datapath=PREPROCESSED_DATAPATH):
    """
    This function loads the test data from the preprocessed_data directory.

    Inputs:
    - schema: The schema of the model (rnn_lw or rnn_sw)
    - input: The type of input data (dynamical or optical)
    - preprocessed_datapath: The path to the preprocessed data directory

    Returns:
    - test_x: The test data in the form of a TensorFlow tensor
    - test_aux_x: The auxiliary test data in the form of a TensorFlow tensor
    - test_y: The test labels in the form of a TensorFlow tensor
    """
    with open(os.path.join(preprocessed_datapath, schema, inputs, "test_x.pt"), "rb") as f:
        test_x = torch.load(f)
    with open(os.path.join(preprocessed_datapath, schema, inputs, "test_aux_x.pt"), "rb") as f:
        test_aux_x = torch.load(f)
    with open(os.path.join(preprocessed_datapath, schema, inputs, "test_y.pt"), "rb") as f:
        test_y = torch.load(f)

    # Convert to TensorFlow tensors
    test_x = convert_pt_to_tf_tensor(test_x)
    test_aux_x = convert_pt_to_tf_tensor(test_aux_x)
    test_y = convert_pt_to_tf_tensor(test_y)

    return test_x, test_aux_x, test_y

def convert_pt_to_tf_tensor(pt_tensor):
    """
    This function takes a PyTorch tensor and converts it to a TensorFlow tensor.
    """
    np_tensor = pt_tensor.numpy()
    tf_tensor = tf.convert_to_tensor(np_tensor)
    return tf_tensor

def load_and_compile_model(model_path, base_lr=BASE_LR, max_lr=MAX_LR, steps_per_epoch=1000):
    """
    This function loads a model from a given path and compiles it.
    (Note: model has to be loaded and compiled in such a way as it uses custom metrics)
    """
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