"""
For a set of models specified by Job ID, this script loads model predictions and test data, and scores
the model. The scores are then saved to a file in a directory specified by the SAVEPATH variable.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    load_test_data,
    err_flux,
    err_derivative,
    rmse_flux,
    rmse_derivative
)

####################################################################################################

ANALYSIS_PATH = "/home/ucaptp0/oasis-rt-surrogate/analysis/trained-models"
PREPROCESSED_DATAPATH = "/home/ucaptp0/oasis-rt-surrogate/data/preprocessed_data"

####################################################################################################

# Load job_id_info and checkpoint_files
with open(os.path.join(ANALYSIS_PATH, "info", "job_id_info.json"), "r") as file:
    job_id_info = json.loads(file.read())

with open(os.path.join(ANALYSIS_PATH, "info", "checkpoint_files.json"), "r") as file:
    checkpoint_files = json.loads(file.read())

# Parse input job_ids from command line
parser = argparse.ArgumentParser(description='Score models')
parser.add_argument('job_ids', metavar='job_ids', type=int, nargs='+',
                    help='Job IDs to score')
args = parser.parse_args()
job_ids = args.job_ids

metrics = [rmse_flux, rmse_derivative, err_flux, err_derivative]
scores = {}
for jid in job_ids:
    schema, inputs = job_id_info[str(jid)]

    # Load targets
    _, _, test_y = load_test_data(schema=schema, inputs=inputs, preprocessed_datapath=PREPROCESSED_DATAPATH)
    
    # Load model predictions
    preds = np.load(os.path.join(ANALYSIS_PATH, "predictions", f"predictions-{jid}.npy"))

    # Calculate scores
    scores[jid] = {}
    for metric in metrics:
        metric_vals0, metric_vals1, metric_vals = metric(test_y, preds)

        # Score
        metric_mean = np.mean(metric_vals)
        metric_std = np.std(metric_vals)
        metric_max = np.max(metric_vals)
        metric_min = np.min(metric_vals)
        scores[jid][metric.__name__ + "_mean"] = metric_mean
        scores[jid][metric.__name__ + "_std"] = metric_std
        scores[jid][metric.__name__ + "_max"] = metric_max
        scores[jid][metric.__name__ + "_min"] = metric_min
        print("metric_vals shape: ", metric_vals.shape, flush=True)
        
        # Plot best, median, worst predictions, and save to file
        if metric.__name__ == "rmse_flux":
            mid_idx = len(metric_vals)//2

            best_flux = [np.argmin(metric_vals0), np.argmin(metric_vals1)]
            worst_flux = [np.argmax(metric_vals0), np.argmax(metric_vals1)]
            median_flux = [np.argsort(metric_vals0)[mid_idx], np.argsort(metric_vals1)[mid_idx]]

        if metric.__name__ == "rmse_derivative":
            mid_idx = len(metric_vals)//2

            best_derivative = [np.argmin(metric_vals0), np.argmin(metric_vals1)]
            worst_derivative = [np.argmax(metric_vals0), np.argmax(metric_vals1)]
            median_derivative = [np.argsort(metric_vals0)[mid_idx], np.argsort(metric_vals1)[mid_idx]]

    if not os.path.exists(os.path.join(ANALYSIS_PATH, "plots", str(jid))):
        os.makedirs(os.path.join(ANALYSIS_PATH, "plots", str(jid)))

    if not os.path.exists(os.path.join(ANALYSIS_PATH, "plots", str(jid), "log_xscale")):
        os.makedirs(os.path.join(ANALYSIS_PATH, "plots", str(jid), "log_xscale"))

    # Plot median flux
    labels = ["Downward Flux", "Upward Flux"]
    fig, axs = plt.subplots(1,2)
    for tgt in range(2):
        axs[tgt].plot(test_y[median_flux[tgt]][:, tgt], range(50), 'g', label="Truth")
        axs[tgt].plot(preds[median_flux[tgt]][:, tgt], range(50), 'r--', label="Prediction")
        axs[tgt].legend()
        axs[tgt].set_title(labels[tgt])
        axs[tgt].set_xlabel("Scaled Flux")
        axs[0].set_ylabel("Level")
    plt.suptitle("Median Flux Prediction")
    plt.savefig(os.path.join(ANALYSIS_PATH, "plots", str(jid), "median_flux-{}.png".format(jid)))
    plt.close()

    # Plot median flux (log xscale)
    fig, axs = plt.subplots(1,2)
    for tgt in range(2):
        axs[tgt].plot(test_y[median_flux[tgt]][:, tgt], range(50), 'g', label="Truth")
        axs[tgt].plot(preds[median_flux[tgt]][:, tgt], range(50), 'r--', label="Prediction")
        axs[tgt].legend()
        axs[tgt].set_title(labels[tgt])
        axs[tgt].set_xlabel("Scaled Flux")
        axs[tgt].set_xscale("log")
        axs[0].set_ylabel("Level")
    plt.suptitle("Median Flux Prediction (log xscale)")
    plt.savefig(os.path.join(ANALYSIS_PATH, "plots", str(jid), "log_xscale", "median_flux_log-{}.png".format(jid)))
    plt.close()

    # Plot median derivative
    labels = ["Downward Flux", "Upward Flux"]
    fig, axs = plt.subplots(1,2)
    for tgt in range(2):
        axs[tgt].plot(test_y[median_derivative[tgt]][:, tgt], range(50), 'g', label="Truth")
        axs[tgt].plot(preds[median_derivative[tgt]][:, tgt], range(50), 'r--', label="Prediction")
        axs[tgt].legend()
        axs[tgt].set_title(labels[tgt])
        axs[tgt].set_xlabel("Scaled Flux")
        axs[0].set_ylabel("Level")
    plt.suptitle("Median Derivative Prediction")
    plt.savefig(os.path.join(ANALYSIS_PATH, "plots", str(jid), "median_derivative-{}.png".format(jid)))
    plt.close()

    # Plot median derivative (log xscale)
    fig, axs = plt.subplots(1,2)
    for tgt in range(2):
        axs[tgt].plot(test_y[median_derivative[tgt]][:, tgt], range(50), 'g', label="Truth")
        axs[tgt].plot(preds[median_derivative[tgt]][:, tgt], range(50), 'r--', label="Prediction")
        axs[tgt].legend()
        axs[tgt].set_title(labels[tgt])
        axs[tgt].set_xlabel("Scaled Flux")
        axs[tgt].set_xscale("log")
        axs[0].set_ylabel("Level")
    plt.suptitle("Median Derivative Prediction (log xscale)")
    plt.savefig(os.path.join(ANALYSIS_PATH, "plots", str(jid), "log_xscale", "median_derivative_log-{}.png".format(jid)))
    plt.close()

    # Best flux
    labels = ["Downward Flux", "Upward Flux"]
    fig, axs = plt.subplots(1,2)
    for tgt in range(2):
        axs[tgt].plot(test_y[best_flux[tgt]][:, tgt], range(50), 'g', label="Truth")
        axs[tgt].plot(preds[best_flux[tgt]][:, tgt], range(50), 'r--', label="Prediction")
        axs[tgt].legend()
        axs[tgt].set_title(labels[tgt])
        axs[tgt].set_xlabel("Scaled Flux")
        axs[0].set_ylabel("Level")
    plt.suptitle("Best Flux Prediction")
    plt.savefig(os.path.join(ANALYSIS_PATH, "plots", str(jid), "best_flux-{}.png".format(jid)))
    plt.close()

    # Best flux (log xscale)
    fig, axs = plt.subplots(1,2)
    for tgt in range(2):
        axs[tgt].plot(test_y[best_flux[tgt]][:, tgt], range(50), 'g', label="Truth")
        axs[tgt].plot(preds[best_flux[tgt]][:, tgt], range(50), 'r--', label="Prediction")
        axs[tgt].legend()
        axs[tgt].set_title(labels[tgt])
        axs[tgt].set_xlabel("Scaled Flux")
        axs[tgt].set_xscale("log")
        axs[0].set_ylabel("Level")
    plt.suptitle("Best Flux Prediction (log xscale)")
    plt.savefig(os.path.join(ANALYSIS_PATH, "plots", str(jid), "log_xscale", "best_flux_log-{}.png".format(jid)))
    plt.close()

    # Worst flux
    fig, axs = plt.subplots(1,2)
    for tgt in range(2):
        axs[tgt].plot(test_y[worst_flux[tgt]][:, tgt], range(50), 'g', label="Truth")
        axs[tgt].plot(preds[worst_flux[tgt]][:, tgt], range(50), 'r--', label="Prediction")
        axs[tgt].legend()
        axs[tgt].set_title(labels[tgt])
        axs[tgt].set_xlabel("Scaled Flux")
        axs[0].set_ylabel("Level")
    plt.suptitle("Worst Flux Prediction")
    plt.savefig(os.path.join(ANALYSIS_PATH, "plots", str(jid), "worst_flux-{}.png".format(jid)))
    plt.close()

    # Worst flux (log xscale)
    fig, axs = plt.subplots(1,2)
    for tgt in range(2):
        axs[tgt].plot(test_y[worst_flux[tgt]][:, tgt], range(50), 'g', label="Truth")
        axs[tgt].plot(preds[worst_flux[tgt]][:, tgt], range(50), 'r--', label="Prediction")
        axs[tgt].legend()
        axs[tgt].set_title(labels[tgt])
        axs[tgt].set_xlabel("Scaled Flux")
        axs[tgt].set_xscale("log")
        axs[0].set_ylabel("Level")
    plt.suptitle("Worst Flux Prediction (log xscale)")
    plt.savefig(os.path.join(ANALYSIS_PATH, "plots", str(jid), "log_xscale", "worst_flux_log-{}.png".format(jid)))
    plt.close()

    # Best derivative
    fig, axs = plt.subplots(1,2)
    for tgt in range(2):
        axs[tgt].plot(test_y[best_derivative[tgt]][:, tgt], range(50), 'g', label="Truth")
        axs[tgt].plot(preds[best_derivative[tgt]][:, tgt], range(50), 'r--', label="Prediction")
        axs[tgt].legend()
        axs[tgt].set_title(labels[tgt])
        axs[tgt].set_xlabel("Scaled Flux")
        axs[0].set_ylabel("Level")
    plt.suptitle("Best Derivative Prediction")
    plt.savefig(os.path.join(ANALYSIS_PATH, "plots", str(jid), "best_derivative-{}.png".format(jid)))
    plt.close()

    # Best derivative (log xscale)
    fig, axs = plt.subplots(1,2)
    for tgt in range(2):
        axs[tgt].plot(test_y[best_derivative[tgt]][:, tgt], range(50), 'g', label="Truth")
        axs[tgt].plot(preds[best_derivative[tgt]][:, tgt], range(50), 'r--', label="Prediction")
        axs[tgt].legend()
        axs[tgt].set_title(labels[tgt])
        axs[tgt].set_xlabel("Scaled Flux")
        axs[tgt].set_xscale("log")
        axs[0].set_ylabel("Level")
    plt.suptitle("Best Derivative Prediction (log xscale)")
    plt.savefig(os.path.join(ANALYSIS_PATH, "plots", str(jid), "log_xscale", "best_derivative_log-{}.png".format(jid)))
    plt.close()

    # Worst derivative
    fig, axs = plt.subplots(1,2)
    for tgt in range(2):
        axs[tgt].plot(test_y[worst_derivative[tgt]][:, tgt], range(50), 'g', label="Truth")
        axs[tgt].plot(preds[worst_derivative[tgt]][:, tgt], range(50), 'r--', label="Prediction")
        axs[tgt].legend()
        axs[tgt].set_title(labels[tgt])
        axs[tgt].set_xlabel("Scaled Flux")
        axs[0].set_ylabel("Level")
    plt.suptitle("Worst Derivative Prediction")
    plt.savefig(os.path.join(ANALYSIS_PATH, "plots", str(jid), "worst_derivative-{}.png".format(jid)))
    plt.close()

    # Worst derivative (log xscale)
    fig, axs = plt.subplots(1,2)
    for tgt in range(2):
        axs[tgt].plot(test_y[worst_derivative[tgt]][:, tgt], range(50), 'g', label="Truth")
        axs[tgt].plot(preds[worst_derivative[tgt]][:, tgt], range(50), 'r--', label="Prediction")
        axs[tgt].legend()
        axs[tgt].set_title(labels[tgt])
        axs[tgt].set_xlabel("Scaled Flux")
        axs[tgt].set_xscale("log")
        axs[0].set_ylabel("Level")
    plt.suptitle("Worst Derivative Prediction (log xscale)")
    plt.savefig(os.path.join(ANALYSIS_PATH, "plots", str(jid), "log_xscale", "worst_derivative_log-{}.png".format(jid)))
    plt.close()

    print(scores[jid])

# Save scores to txt file
with open(os.path.join(ANALYSIS_PATH, "info", "scores.txt"), "w") as f:
    f.write(str(scores))
