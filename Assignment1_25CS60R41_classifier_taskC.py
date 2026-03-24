#!/usr/bin/env python3
# messy beginner style version of task C

from pathlib import Path
import csv
import numpy as np

base_folder = Path(".")


def my_sigmoid(arr):

    clipped = np.clip(arr, -40.0, 40.0)

    res = 1.0 / (1.0 + np.exp(-clipped))

    return res


def calc_log_losses(w, X, y):

    scores = X @ w

    probs = my_sigmoid(scores)

    eps = 1e-12

    losses = -(y * np.log(probs + eps) + (1.0 - y) * np.log(1.0 - probs + eps))

    return losses


def loss_grad(w, X, y):

    losses = calc_log_losses(w, X, y)

    scores = X @ w

    probs = my_sigmoid(scores)

    grad = (X.T @ (probs - y)) / X.shape[0]

    avg = float(np.mean(losses))

    return avg, grad


def add_bias(X):

    n = X.shape[0]

    ones = np.ones((n, 1), dtype=float)

    newX = np.hstack([ones, X])

    return newX


def predict_labels(w, X):

    scores = X @ w

    probs = my_sigmoid(scores)

    preds = (probs >= 0.5).astype(int)

    return preds


def accuracy_score(y_true, y_pred):

    val = np.mean(y_true == y_pred)

    return float(val)


def cov_measure(w, X, z):

    z_mean = np.mean(z)

    centered = z - z_mean

    scores = X @ w

    val = np.mean(centered * scores)

    return float(val)


def p_rule(pred, z):

    mask0 = z == 0
    mask1 = z == 1

    rate0 = 0.0
    rate1 = 0.0

    if np.any(mask0):
        rate0 = float(pred[mask0].mean())

    if np.any(mask1):
        rate1 = float(pred[mask1].mean())

    eps = 1e-12

    if rate0 <= eps and rate1 <= eps:
        return 100.0

    if rate0 <= eps or rate1 <= eps:
        return 0.0

    val = min(rate1 / rate0, rate0 / rate1) * 100.0

    return float(val)


def train_basic(X, y, iters=2500, lr=0.1, reg=1e-4):

    w = np.zeros(X.shape[1], dtype=float)

    for i in range(iters):

        loss_val, grad = loss_grad(w, X, y)

        step = grad + reg * w

        w = w - lr * step

    return w


def read_csv_file(path):

    arr = np.genfromtxt(path, delimiter=",", names=True)

    x1 = arr["x1"]
    x2 = arr["x2"]

    X = np.column_stack([x1, x2]).astype(float)

    z = arr["z"].astype(int)

    y_tmp = arr["y"].astype(int)

    y = ((y_tmp + 1) // 2).astype(int)

    return X, y, z


def save_rows(rows, path, headers):

    path.parent.mkdir(parents=True, exist_ok=True)

    f = path.open("w", newline="", encoding="utf-8")

    writer = csv.writer(f)

    writer.writerow(headers)

    for r in rows:

        row = []

        for h in headers:
            row.append(r[h])

        writer.writerow(row)

    f.close()


def fairness_val_grad(w, X, z):

    z_mean = np.mean(z)

    centered = z - z_mean

    scores = X @ w

    cov_val = np.mean(centered * scores)

    cov_grad = np.mean(centered[:, None] * X, axis=0)

    val = cov_val * cov_val

    grad = 2.0 * cov_val * cov_grad

    return float(val), grad


def train_fair_task_c(X, y, z, gamma, iters=1500, lr_w=0.05, lr_lambda=0.1, reg=1e-4, lambda_cap=200.0):

    theta_star = train_basic(X, y, iters=2500, lr=0.1, reg=reg)

    loss_star = calc_log_losses(theta_star, X, y)

    rhs = (1.0 + gamma) * loss_star

    n = X.shape[0]

    w = theta_star.copy()

    lambda_vec = np.zeros(n, dtype=float)

    z_mean = np.mean(z)

    z_center = z - z_mean

    cov_grad = np.mean(z_center[:, None] * X, axis=0)

    for i in range(iters):

        scores = X @ w

        probs = my_sigmoid(scores)

        eps = 1e-12

        loss = -(y * np.log(probs + eps) + (1.0 - y) * np.log(1.0 - probs + eps))

        violation = loss - rhs

        cov_val = np.mean(z_center * scores)

        grad_fair = (cov_val / np.sqrt(cov_val * cov_val + 1e-12)) * cov_grad

        dloss = ((probs - y)[:, None] * X)

        grad_cons = np.mean((lambda_vec[:, None] * dloss), axis=0)

        step = grad_fair + grad_cons + reg * w

        w = w - lr_w * step

        lambda_vec = lambda_vec + lr_lambda * violation

        lambda_vec = np.maximum(0.0, lambda_vec)

        lambda_vec = np.minimum(lambda_cap, lambda_vec)

    return w


# -------------------------------

gamma_values = [0.0, 0.5, 1.0, 1.5]

taskA_split = base_folder / "TaskA" / "data" / "splits"
taskC_results = base_folder / "TaskC" / "results"

taskC_results.mkdir(parents=True, exist_ok=True)

datasets = {}

names = ["phi_pi", "phi_pi_2", "phi_pi_4", "phi_pi_6", "phi_pi_8"]

for name in names:

    tr_path = taskA_split / "synthetic" / (name + "_train.csv")
    te_path = taskA_split / "synthetic" / (name + "_test.csv")

    x_tr, y_tr, z_tr = read_csv_file(tr_path)
    x_te, y_te, z_te = read_csv_file(te_path)

    datasets[name] = (x_tr, y_tr, z_tr, x_te, y_te, z_te)


adult_tr = np.load(taskA_split / "adult" / "adult_train.npz")
adult_te = np.load(taskA_split / "adult" / "adult_test.npz")

datasets["adult"] = (
    adult_tr["x"],
    adult_tr["y"],
    adult_tr["z"],
    adult_te["x"],
    adult_te["y"],
    adult_te["z"],
)

all_rows = []

for name, data in datasets.items():

    x_tr, y_tr, z_tr, x_te, y_te, z_te = data

    x_tr_b = add_bias(x_tr)
    x_te_b = add_bias(x_te)

    local_rows = []

    for g in gamma_values:

        w_now = train_fair_task_c(x_tr_b, y_tr, z_tr, g)

        preds = predict_labels(w_now, x_te_b)

        acc = accuracy_score(y_te, preds)

        cov = abs(cov_measure(w_now, x_te_b, z_te))

        pval = p_rule(preds, z_te)

        row = {
            "dataset": name,
            "method": "taskC_fair",
            "gamma": float(g),
            "test_accuracy": acc,
            "test_covariance": cov,
            "test_p_rule": pval,
        }

        local_rows.append(row)

    for r in local_rows:
        all_rows.append(r)

    save_rows(
        local_rows,
        taskC_results / (name + "_taskC.csv"),
        ["dataset", "method", "gamma", "test_accuracy", "test_covariance", "test_p_rule"],
    )

save_rows(
    all_rows,
    taskC_results / "taskC_all_datasets.csv",
    ["dataset", "method", "gamma", "test_accuracy", "test_covariance", "test_p_rule"],
)

print("Task C done. total rows:", len(all_rows))