#!/usr/bin/env python3
# messy beginner version for task D

from pathlib import Path
import csv
import math
import numpy as np

base_dir = Path(".")


def sigmoid_fun(vals):

    clipped = np.clip(vals, -40.0, 40.0)

    res = 1.0 / (1.0 + np.exp(-clipped))

    return res


def calc_losses(w, X, y):

    scores = X @ w

    probs = sigmoid_fun(scores)

    eps = 1e-12

    loss_vals = -(y * np.log(probs + eps) + (1.0 - y) * np.log(1.0 - probs + eps))

    return loss_vals


def loss_grad(w, X, y):

    losses = calc_losses(w, X, y)

    scores = X @ w

    probs = sigmoid_fun(scores)

    grad = (X.T @ (probs - y)) / X.shape[0]

    avg_loss = float(np.mean(losses))

    return avg_loss, grad


def add_bias(X):

    n = X.shape[0]

    ones = np.ones((n, 1))

    X2 = np.hstack([ones, X])

    return X2


def predict_now(w, X):

    scores = X @ w

    probs = sigmoid_fun(scores)

    preds = (probs >= 0.5).astype(int)

    return preds


def accuracy_val(y_true, y_pred):

    val = np.mean(y_true == y_pred)

    return float(val)


def cov_val(w, X, z):

    z_mean = np.mean(z)

    centered = z - z_mean

    scores = X @ w

    val = np.mean(centered * scores)

    return float(val)


def p_rule(pred, z):

    mask0 = z == 0
    mask1 = z == 1

    r0 = 0.0
    r1 = 0.0

    if np.any(mask0):
        r0 = float(pred[mask0].mean())

    if np.any(mask1):
        r1 = float(pred[mask1].mean())

    eps = 1e-12

    if r0 <= eps and r1 <= eps:
        return 100.0

    if r0 <= eps or r1 <= eps:
        return 0.0

    val = min(r1 / r0, r0 / r1) * 100.0

    return float(val)


def train_basic(X, y, iters=2500, lr=0.1, reg=1e-4):

    w = np.zeros(X.shape[1])

    for i in range(iters):

        loss_val, grad = loss_grad(w, X, y)

        step = grad + reg * w

        w = w - lr * step

    return w


def read_split(path):

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


def rot_matrix(angle):

    c = math.cos(angle)

    s = math.sin(angle)

    mat = np.array([[c, -s], [s, c]])

    return mat


def gaussian_batch(X, mean, cov):

    inv_cov = np.linalg.inv(cov)

    det_cov = np.linalg.det(cov)

    dim = mean.shape[0]

    diff = X - mean

    quad = np.einsum("bi,ij,bj->b", diff, inv_cov, diff)

    norm = 1.0 / math.sqrt(((2.0 * math.pi) ** dim) * det_cov)

    res = norm * np.exp(-0.5 * quad)

    return res


def make_split(n, ratio, seed):

    rng = np.random.default_rng(seed)

    idx = np.arange(n)

    rng.shuffle(idx)

    test_n = int(round(n * ratio))

    train_idx = idx[test_n:]

    test_idx = idx[:test_n]

    return train_idx, test_idx


def project_cov_limit(w, g, c):

    g2 = float(np.dot(g, g))

    if g2 < 1e-14:
        return w

    val = float(np.dot(g, w))

    if val > c:
        w = w - ((val - c) / g2) * g

    elif val < -c:
        w = w - ((val + c) / g2) * g

    return w


def train_taskB(X, y, z, c_limit, iters=3000, lr=0.08, reg=1e-4):

    w = train_basic(X, y, iters=1000, lr=0.08, reg=reg)

    z_mean = np.mean(z)

    z_center = z - z_mean

    g = np.mean(z_center[:, None] * X, axis=0)

    for i in range(iters):

        loss_val, grad = loss_grad(w, X, y)

        w = w - lr * (grad + reg * w)

        w = project_cov_limit(w, g, c_limit)

    return w


def train_taskC(X, y, z, gamma, iters=1500, lr_w=0.05, lr_l=0.1, reg=1e-4, cap=200.0):

    theta_star = train_basic(X, y)

    loss_star = calc_losses(theta_star, X, y)

    rhs = (1.0 + gamma) * loss_star

    n = X.shape[0]

    w = theta_star.copy()

    lamb = np.zeros(n)

    z_center = z - np.mean(z)

    cov_grad = np.mean(z_center[:, None] * X, axis=0)

    for i in range(iters):

        scores = X @ w

        probs = sigmoid_fun(scores)

        eps = 1e-12

        loss = -(y * np.log(probs + eps) + (1.0 - y) * np.log(1.0 - probs + eps))

        violation = loss - rhs

        cov_val = np.mean(z_center * scores)

        grad_fair = (cov_val / np.sqrt(cov_val * cov_val + 1e-12)) * cov_grad

        dloss = ((probs - y)[:, None] * X)

        grad_cons = np.mean((lamb[:, None] * dloss), axis=0)

        step = grad_fair + grad_cons + reg * w

        w = w - lr_w * step

        lamb = lamb + lr_l * violation

        lamb = np.maximum(0.0, lamb)

        lamb = np.minimum(cap, lamb)

    return w


def make_noisy_data(n, sigma, seed):

    rng = np.random.default_rng(seed)

    mean_pos = np.array([1.0, -3.0])
    cov_pos = np.array([[3.0, 1.0], [1.0, 2.0]])

    mean_neg = np.array([-3.0, 2.0])
    cov_neg = np.array([[7.0, 1.0], [1.0, 10.0]])

    y_true = rng.choice([-1, 1], size=n)

    X = np.zeros((n, 2))

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == -1)[0]

    if pos_idx.size > 0:
        X[pos_idx] = rng.multivariate_normal(mean_pos, cov_pos, pos_idx.size)

    if neg_idx.size > 0:
        X[neg_idx] = rng.multivariate_normal(mean_neg, cov_neg, neg_idx.size)

    p_pos = gaussian_batch(X, mean_pos, cov_pos)
    p_neg = gaussian_batch(X, mean_neg, cov_neg)

    log_odds = np.log((p_pos + 1e-12) / (p_neg + 1e-12))

    eps_noise = rng.normal(0.0, sigma, size=n)

    y_noisy = np.where(log_odds + eps_noise >= 0.0, 1, -1)

    X_rot = X @ rot_matrix(math.pi / 4).T

    p_pos2 = gaussian_batch(X_rot, mean_pos, cov_pos)
    p_neg2 = gaussian_batch(X_rot, mean_neg, cov_neg)

    pz = p_pos2 / (p_pos2 + p_neg2 + 1e-12)

    z = rng.binomial(1, pz)

    return X_rot, z, y_true, y_noisy


# ------------------------

N = 10000
sigmas = [0.4, 0.7]
seed = 42
test_ratio = 0.2

result_dir = base_dir / "TaskD" / "results"

result_dir.mkdir(parents=True, exist_ok=True)

rows = []

for s in sigmas:

    X, z, y_true_pm, y_noisy_pm = make_noisy_data(N, s, seed)

    y_true = ((y_true_pm + 1) // 2)
    y_noisy = ((y_noisy_pm + 1) // 2)

    tr, te = make_split(N, test_ratio, seed + int(1000 * s))

    Xtr = X[tr]
    Xte = X[te]

    ztr = z[tr]
    zte = z[te]

    y_noisy_tr = y_noisy[tr]
    y_noisy_te = y_noisy[te]

    y_true_te = y_true[te]

    Xtr_b = add_bias(Xtr)
    Xte_b = add_bias(Xte)

    wB = train_taskB(Xtr_b, y_noisy_tr, ztr, c_limit=0.0)
    predB = predict_now(wB, Xte_b)

    wC = train_taskC(Xtr_b, y_noisy_tr, ztr, gamma=0.0)
    predC = predict_now(wC, Xte_b)

    for method, w, pred in [
        ("taskB_c0", wB, predB),
        ("taskC_gamma0", wC, predC),
    ]:

        row = {
            "sigma": float(s),
            "method": method,
            "test_accuracy_noisy": accuracy_val(y_noisy_te, pred),
            "test_accuracy_true": accuracy_val(y_true_te, pred),
            "test_covariance": abs(cov_val(w, Xte_b, zte)),
            "test_p_rule": p_rule(pred, zte),
        }

        rows.append(row)

save_rows(
    rows,
    result_dir / "taskD_results.csv",
    [
        "sigma",
        "method",
        "test_accuracy_noisy",
        "test_accuracy_true",
        "test_covariance",
        "test_p_rule",
    ],
)

print("Task D done, rows:", len(rows))