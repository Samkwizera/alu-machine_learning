#!/usr/bin/env python3
"""
Bayesian Optimization with GPyOpt
File: 6-bayes_opt.py
Directory: unsupervised_learning/hyperparameter_tuning
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import GPyOpt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


os.makedirs("checkpoints", exist_ok=True)

# Load MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape((-1, 28 * 28))
x_test = x_test.reshape((-1, 28 * 28))

# Use a smaller subset to make optimization faster
x_train = x_train[:12000]
y_train = y_train[:12000]

x_val = x_test[:3000]
y_val = y_test[:3000]

TARGET_ACCURACY = 0.97
results = []


def build_model(learning_rate, units, dropout_rate, l2_weight):
    """
    Builds a simple neural network for MNIST classification.
    """
    model = keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_weight)
        ),
        layers.Dropout(dropout_rate),
        layers.Dense(
            units // 2,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_weight)
        ),
        layers.Dropout(dropout_rate),
        layers.Dense(10, activation="softmax")
    ])

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def objective_function(x):
    """
    Objective function for Bayesian Optimization.

    GPyOpt minimizes this function, so we return:
    1 - best validation accuracy

    The single satisficing metric is validation accuracy.
    """
    learning_rate = float(x[:, 0][0])
    units = int(x[:, 1][0])
    dropout_rate = float(x[:, 2][0])
    l2_weight = float(x[:, 3][0])
    batch_size = int(x[:, 4][0])

    checkpoint_name = (
        f"checkpoints/best_lr-{learning_rate:.5f}"
        f"_units-{units}"
        f"_dropout-{dropout_rate:.2f}"
        f"_l2-{l2_weight:.6f}"
        f"_batch-{batch_size}.keras"
    )

    model = build_model(
        learning_rate=learning_rate,
        units=units,
        dropout_rate=dropout_rate,
        l2_weight=l2_weight
    )

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_name,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=0
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        mode="max",
        restore_best_weights=True
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=20,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stopping],
        verbose=0
    )

    best_val_accuracy = max(history.history["val_accuracy"])
    objective_value = 1.0 - best_val_accuracy

    run_result = {
        "learning_rate": learning_rate,
        "units": units,
        "dropout_rate": dropout_rate,
        "l2_weight": l2_weight,
        "batch_size": batch_size,
        "best_val_accuracy": float(best_val_accuracy),
        "objective_value": float(objective_value),
        "checkpoint": checkpoint_name
    }

    results.append(run_result)

    print(json.dumps(run_result, indent=4))

    return objective_value


domain = [
    {
        "name": "learning_rate",
        "type": "continuous",
        "domain": (0.0001, 0.01)
    },
    {
        "name": "units",
        "type": "discrete",
        "domain": (64, 128, 256, 512)
    },
    {
        "name": "dropout_rate",
        "type": "continuous",
        "domain": (0.1, 0.6)
    },
    {
        "name": "l2_weight",
        "type": "continuous",
        "domain": (0.000001, 0.01)
    },
    {
        "name": "batch_size",
        "type": "discrete",
        "domain": (32, 64, 128, 256)
    }
]


optimizer = GPyOpt.methods.BayesianOptimization(
    f=objective_function,
    domain=domain,
    model_type="GP",
    acquisition_type="EI",
    exact_feval=False,
    maximize=False
)

optimizer.run_optimization(max_iter=30)

# Plot convergence
optimizer.plot_convergence()
plt.savefig("bayes_opt_convergence.png")

best_index = int(np.argmin(optimizer.Y))
best_params = optimizer.X[best_index]
best_score = optimizer.Y[best_index][0]

report = {
    "satisficing_metric": "validation_accuracy",
    "target_accuracy": TARGET_ACCURACY,
    "best_objective_value": float(best_score),
    "best_validation_accuracy": float(1.0 - best_score),
    "best_hyperparameters": {
        "learning_rate": float(best_params[0]),
        "units": int(best_params[1]),
        "dropout_rate": float(best_params[2]),
        "l2_weight": float(best_params[3]),
        "batch_size": int(best_params[4])
    },
    "all_results": results
}

with open("bayes_opt.txt", "w") as f:
    f.write("Bayesian Optimization Report\n")
    f.write("============================\n\n")
    f.write(json.dumps(report, indent=4))

print("\nOptimization complete.")
print("Report saved to bayes_opt.txt")
print("Convergence plot saved to bayes_opt_convergence.png")
