"""
This file contains code to train an RNN model as prescribed in
Ukkonen et al 2022 (https://doi.org/10.1029/2021MS002875).
"""

import os
import sys
import time
import torch
import datetime
import tensorflow as tf
import tensorflow_addons as tfa
from sys import platform
from tensorflow import keras
from tensorflow.keras import losses, optimizers, layers, Input, Model
from tensorflow.keras.layers import GRU, Dense, TimeDistributed, Lambda
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger

from model.rnn_sw.utils_sw import (
    custom_loss,
    load_input_data,
    load_aux_data,
    convert_pt_to_tf_tensor,
    rmse_flux,
    rmse_derivative
)

from model.rnn_sw.vars_sw import *

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], "GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialised
        print(e)

############################# INPUT DATA #############################

TIME = time.time()
DATAPATH = os.path.join(DATAPATH, "processed", "opacities", "cosz_as_vector_input") 
datapath = DATAPATH
print(datapath, flush=True)

CHECKPOINT_PATH = os.path.join(
    DATAPATH, f"checkpoints-{TIME}", "{epoch:02d}-{val_loss:.2f}/checkpoint.model.keras"
)

checkpoint_path = CHECKPOINT_PATH

train_x, train_y, val_x, val_y = load_input_data(datapath=datapath)

(
    train_x,
    train_y,
    val_x,
    val_y,
) = [
    convert_pt_to_tf_tensor(tens)
    for tens in (
        train_x,
        train_y,
        val_x,
        val_y,
    )
]

nvar = train_x.shape[-1]  # Number of features
nlev = train_x.shape[1]
nlay = nlev - 1
noutputvar = train_y.shape[-1]  # Should be 2 for up and down flux, shortwave only

inputs = Input(shape=(nlev, nvar), name="inputs_main")
targets = Input(shape=(nlev, noutputvar), name="targets")

########################### INPUT VARIABLES ##########################

# Activation Functions
activ_RNN1 = "relu"
activ_Dense = "relu"
activ_RNN2 = "relu"
activ_RNN3 = "relu"
activ_output = "sigmoid"

epochs = EPOCHS
patience = PATIENCE
batch_size = BATCH_SIZE

base_lr = BASE_LR
max_lr = MAX_LR

alpha = float(sys.argv[1])
third_rnn = bool(sys.argv[2]) # THIRD_RNN
num_neurons = int(sys.argv[3])

print("Alpha: ", alpha)
print("third_rnn: ", third_rnn)
print("num_neurons: ", num_neurons)
print("base_lr: ", base_lr)
print("max_lr: ", max_lr)
print("batch_size: ", batch_size)
print("patience: ", patience)

steps_per_epoch = train_x.shape[0] // batch_size

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

######################### MODEL ARCHITECTURE #########################

# First RNN
RNN1_outputs, last_state = GRU(
    num_neurons,
    return_sequences=True,
    return_state=True,
    activation=activ_RNN1,
)(inputs)

# Second RNN
RNN2_outputs = GRU(num_neurons, return_sequences=True, go_backwards=True, activation=activ_RNN2)(RNN1_outputs)

BIRNN_outputs = tf.concat([RNN1_outputs, RNN2_outputs], axis=2)


if third_rnn:
    RNN3_inputs = BIRNN_outputs

    # Third RNN
    RNN3_outputs = GRU(num_neurons, return_sequences=True, activation=activ_RNN3)(RNN3_inputs)

    outputs = TimeDistributed(layers.Dense(noutputvar, activation=activ_output), name="dense_output")(
        RNN3_outputs
    )

else:
    outputs = TimeDistributed(layers.Dense(noutputvar, activation=activ_output), name="dense_output")(
        BIRNN_outputs
    )

model = Model(inputs=[inputs, targets], outputs=outputs)
model.add_loss(custom_loss(y_true=targets, y_pred=outputs, alpha=alpha))
model.add_metric(rmse_flux(y_true=targets, y_pred=outputs), name="rmse_flux")
model.add_metric(
    rmse_derivative(y_true=targets, y_pred=outputs), name="rmse_derivative"
)
model.compile(
    optimizer=optim,
    metrics=[
        keras.metrics.MeanAbsolutePercentageError(),
        keras.metrics.MeanSquaredError(),
    ],
)
model.summary()

if platform == "darwin":
    keras.utils.plot_model(model, "rnn.png")

log_dir = "logs/fit/" + str(TIME)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

csv_logger = CSVLogger("training.log")
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=False, save_best_only=True, verbose=1
)
es_callback = EarlyStopping(monitor='val_loss', patience=5)

callbacks = [csv_logger, cp_callback, tensorboard_callback, es_callback]

########################### MODEL TRAINING ###########################

history = model.fit(
    x=[
        train_x,
        train_y,
    ],
    y=train_y,
    epochs=epochs,
    batch_size=batch_size,
    steps_per_epoch=train_x.shape[0] // batch_size,
    shuffle=True,
    verbose=1,
    validation_data=([val_x, val_y], val_y),
    callbacks=callbacks,
)
