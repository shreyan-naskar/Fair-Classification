#!/usr/bin/env python3
# messy beginner style version

from pathlib import Path
import csv
import numpy as np

base = Path(".")


def sigmoid_like(values):

    clipped = np.clip(values, -40.0, 40.0)

    result = 1.0 / (1.0 + np.exp(-clipped))

    return result


def calc_each_loss(w, X, y):

    scores = X @ w

    probs = sigmoid_like(scores)

    eps = 1e-12

    losses = -(y * np.log(probs + eps) + (1.0 - y) * np.log(1.0 - probs + eps))

    return losses


def loss_and_grad(w, X, y):

    losses = calc_each_loss(w, X, y)

    scores = X @ w

    probs = sigmoid_like(scores)

    grad = (X.T @ (probs - y)) / X.shape[0]

    avg_loss = float(np.mean(losses))

    return avg_loss, grad


def put_bias(X):

    n = X.shape[0]

    bias = np.ones((n, 1), dtype=float)

    X2 = np.hstack([bias, X])

    return X2


def predict_now(w, X):

    scores = X @ w

    probs = sigmoid_like(scores)

    preds = (probs >= 0.5).astype(int)

    return preds


def acc_score(y_true, y_pred):

    val = np.mean(y_true == y_pred)

    return float(val)


def cov_value(w, X, z):

    z_mean = np.mean(z)

    centered = z - z_mean

    scores = X @ w

    val = np.mean(centered * scores)

    return float(val)


def p_rule_calc(pred, z):

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


def train_simple(X, y, iters=2500, lr=0.1, reg=1e-4):

    w = np.zeros(X.shape[1], dtype=float)

    for i in range(iters):

        loss_val, grad = loss_and_grad(w, X, y)

        step = grad + reg * w

        w = w - lr * step

    return w


def read_csv_split(path):

    arr = np.genfromtxt(path, delimiter=",", names=True)

    x1 = arr["x1"]
    x2 = arr["x2"]

    X = np.column_stack([x1, x2]).astype(float)

    z = arr["z"].astype(int)

    y_raw = arr["y"].astype(int)

    y = ((y_raw + 1) // 2).astype(int)

    return X, y, z


def write_rows(rows, path, headers):

    path.parent.mkdir(parents=True, exist_ok=True)

    f = path.open("w", newline="", encoding="utf-8")

    w = csv.writer(f)

    w.writerow(headers)

    for r in rows:

        row = []

        for h in headers:
            row.append(r[h])

        w.writerow(row)

    f.close()


def project_limit(w, g, c):

    g_norm = float(np.dot(g, g))

    if g_norm < 1e-14:
        return w

    proj = float(np.dot(g, w))

    if proj > c:

        w = w - ((proj - c) / g_norm) * g

    elif proj < -c:

        w = w - ((proj + c) / g_norm) * g

    return w


def train_fair(X, y, z, c_limit, iters=3000, lr=0.08, reg=1e-4):

    w = train_simple(X, y, iters=1000, lr=0.08, reg=reg)

    z_mean = np.mean(z)

    centered = z - z_mean

    g = np.mean(centered[:, None] * X, axis=0)

    for i in range(iters):

        loss_val, grad = loss_and_grad(w, X, y)

        step = grad + reg * w

        w = w - lr * step

        w = project_limit(w, g, c_limit)

    return w


# ----------------------------

c_list = [0.8, 0.5, 0.2, 0.0]

taskA_split = base / "TaskA" / "data" / "splits"
taskB_res = base / "TaskB" / "results"

taskB_res.mkdir(parents=True, exist_ok=True)

datasets = {}

names = ["phi_pi", "phi_pi_2", "phi_pi_4", "phi_pi_6", "phi_pi_8"]

for name in names:

    tr_path = taskA_split / "synthetic" / (name + "_train.csv")
    te_path = taskA_split / "synthetic" / (name + "_test.csv")

    x_tr, y_tr, z_tr = read_csv_split(tr_path)
    x_te, y_te, z_te = read_csv_split(te_path)

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

    x_tr_b = put_bias(x_tr)
    x_te_b = put_bias(x_te)

    local_rows = []

    for c_val in c_list:

        w_now = train_fair(x_tr_b, y_tr, z_tr, c_val)

        pred = predict_now(w_now, x_te_b)

        acc = acc_score(y_te, pred)

        cov = abs(cov_value(w_now, x_te_b, z_te))

        p_rule = p_rule_calc(pred, z_te)

        row = {
            "dataset": name,
            "method": "taskB_fair",
            "c": float(c_val),
            "test_accuracy": acc,
            "test_covariance": cov,
            "test_p_rule": p_rule,
        }

        local_rows.append(row)

    for r in local_rows:
        all_rows.append(r)

    write_rows(
        local_rows,
        taskB_res / (name + "_taskB.csv"),
        ["dataset", "method", "c", "test_accuracy", "test_covariance", "test_p_rule"],
    )

write_rows(
    all_rows,
    taskB_res / "taskB_all_datasets.csv",
    ["dataset", "method", "c", "test_accuracy", "test_covariance", "test_p_rule"],
)

print("Task B done, rows:", len(all_rows))