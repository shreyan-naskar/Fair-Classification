#!/usr/bin/env python3

from pathlib import Path
import csv
import math
import urllib.request
import numpy as np


base_folder = Path(".")
total_points = 10000
random_seed = 42
test_part = 0.2

phi_values = [
    ("phi_pi", math.pi),
    ("phi_pi_2", math.pi / 2),
    ("phi_pi_4", math.pi / 4),
    ("phi_pi_6", math.pi / 6),
    ("phi_pi_8", math.pi / 8),
]

adult_train_link = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
adult_test_link = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"


def make_rot_matrix(angle):

    c = math.cos(angle)
    s = math.sin(angle)

    mat = np.array([
        [c, -s],
        [s, c]
    ], dtype=float)

    return mat


def gaussian_many(list_points, mean_v, cov_m):

    inv_cov = np.linalg.inv(cov_m)
    det_cov = np.linalg.det(cov_m)

    d = len(mean_v)

    part = ((2 * math.pi) ** d) * det_cov
    norm_const = 1.0 / math.sqrt(part)

    res = []

    for p in list_points:

        diff = p - mean_v
        temp = diff @ inv_cov @ diff.T
        val = norm_const * math.exp(-0.5 * temp)

        res.append(val)

    arr = np.array(res, dtype=float)

    return arr


def create_fake_data(n, phi, seed):

    rng = np.random.default_rng(seed)

    labels = rng.choice([-1, 1], size=n)

    pos_mean = [1.0, -3.0]
    pos_cov = [[3.0, 1.0], [1.0, 2.0]]

    neg_mean = [-3.0, 2.0]
    neg_cov = [[7.0, 1.0], [1.0, 10.0]]

    X = np.zeros((n, 2))

    pos_list = []
    neg_list = []

    for i in range(n):
        if labels[i] == 1:
            pos_list.append(i)
        else:
            neg_list.append(i)

    if len(pos_list) > 0:

        s = rng.multivariate_normal(pos_mean, pos_cov, len(pos_list))

        for j in range(len(pos_list)):
            idx = pos_list[j]
            X[idx] = s[j]

    if len(neg_list) > 0:

        s2 = rng.multivariate_normal(neg_mean, neg_cov, len(neg_list))

        for j in range(len(neg_list)):
            idx = neg_list[j]
            X[idx] = s2[j]

    rot = make_rot_matrix(phi)

    X_rot = np.dot(X, rot.T)

    p1 = gaussian_many(X_rot, pos_mean, pos_cov)
    p2 = gaussian_many(X_rot, neg_mean, neg_cov)

    sens_prob = p1 / (p1 + p2 + 1e-12)

    sens = rng.binomial(1, sens_prob)

    return X_rot, sens.astype(int), labels


def make_train_test_idx(n, ratio, seed):

    rng = np.random.default_rng(seed)

    arr = np.arange(n)

    rng.shuffle(arr)

    test_n = int(round(n * ratio))

    test_idx = arr[:test_n]
    train_idx = arr[test_n:]

    return train_idx, test_idx


def save_csv(file_path, X, Z, Y):

    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)

    f = file_path.open("w", newline="", encoding="utf-8")

    w = csv.writer(f)

    w.writerow(["x1", "x2", "z", "y"])

    for i in range(len(X)):

        a = "{:.10f}".format(X[i][0])
        b = "{:.10f}".format(X[i][1])

        z_val = int(Z[i])
        y_val = int(Y[i])

        w.writerow([a, b, z_val, y_val])

    f.close()


def download_if_needed(path, url):

    if path.exists() == False:
        urllib.request.urlretrieve(url, str(path))


def read_adult(path):

    data = []

    f = path.open("r", encoding="utf-8")

    for line in f:

        line = line.strip()

        if line == "":
            continue

        if line.startswith("|"):
            continue

        parts = line.split(",")

        row = []

        for p in parts:
            row.append(p.strip())

        if len(row) == 15:
            data.append(row)

    f.close()

    return data


def get_mode(values):

    count = {}
    count = {}

    for v in values:
        if v in count:
            count[v] += 1
        else:
            count[v] = 1

    best = None
    best_c = -1

    for k in count:
        if count[k] > best_c:
            best_c = count[k]
            best = k

    return best


def scale_data(trainX, testX):

    mean = trainX.mean(axis=0)
    std = trainX.std(axis=0)

    for i in range(len(std)):
        if std[i] < 1e-12:
            std[i] = 1.0

    train_scaled = (trainX - mean) / std
    test_scaled = (testX - mean) / std

    return train_scaled, test_scaled


def encode_rows(rows, num_cols, cat_cols, vocab):

    nums = []

    for r in rows:

        temp = []

        for c in num_cols:
            temp.append(float(r[c]))

        nums.append(temp)

    X_num = np.array(nums, dtype=float)

    cat_parts = []

    for c in cat_cols:

        cats = vocab[c]

        mp = {}

        for i in range(len(cats)):
            mp[cats[i]] = i

        arr = np.zeros((len(rows), len(cats)))

        for i in range(len(rows)):

            v = rows[i][c]

            if v in mp:
                j = mp[v]
                arr[i][j] = 1

        cat_parts.append(arr)

    if len(cat_parts) > 0:
        X_cat = np.hstack(cat_parts)
    else:
        X_cat = np.empty((len(rows), 0))

    X = np.hstack([X_num, X_cat])

    y_list = []

    for r in rows:
        if r[-1] == ">50K":
            y_list.append(1)
        else:
            y_list.append(0)

    y = np.array(y_list)

    z_list = []

    for r in rows:
        if r[9] == "Male":
            z_list.append(1)
        else:
            z_list.append(0)

    z = np.array(z_list)

    return X, y, z


def make_dirs():

    syn_raw = base_folder / "TaskA" / "data" / "raw" / "synthetic"
    ad_raw = base_folder / "TaskA" / "data" / "raw" / "adult"

    syn_split = base_folder / "TaskA" / "data" / "splits" / "synthetic"
    ad_split = base_folder / "TaskA" / "data" / "splits" / "adult"

    arr = [syn_raw, ad_raw, syn_split, ad_split]

    for p in arr:
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)

    return syn_raw, ad_raw, syn_split, ad_split


def make_synth_files(raw_dir, split_dir):

    for i in range(len(phi_values)):

        name = phi_values[i][0]
        phi = phi_values[i][1]

        X, Z, Y = create_fake_data(total_points, phi, random_seed + i)

        path_full = raw_dir / (name + ".csv")

        save_csv(path_full, X, Z, Y)

        train_i, test_i = make_train_test_idx(total_points, test_part, random_seed + 100 + i)

        Xtr = X[train_i]
        Ztr = Z[train_i]
        Ytr = Y[train_i]

        save_csv(split_dir / (name + "_train.csv"), Xtr, Ztr, Ytr)

        Xte = X[test_i]
        Zte = Z[test_i]
        Yte = Y[test_i]

        save_csv(split_dir / (name + "_test.csv"), Xte, Zte, Yte)

    print("done synthetic datasets")


def make_adult_npz(raw_dir, split_dir):

    download_if_needed(raw_dir / "adult.data", adult_train_link)
    download_if_needed(raw_dir / "adult.test", adult_test_link)

    train_rows = read_adult(raw_dir / "adult.data")
    test_rows = read_adult(raw_dir / "adult.test")

    for r in test_rows:
        r[-1] = r[-1].replace(".", "")

    num_cols = [0,2,4,10,11,12]
    cat_cols = [1,3,5,6,7,8,9,13]

    mode_map = {}

    for c in cat_cols:

        vals = []

        for r in train_rows:
            if r[c] != "?":
                vals.append(r[c])

        mode_map[c] = get_mode(vals)

    for rows in [train_rows, test_rows]:

        for r in rows:

            for c in cat_cols:

                if r[c] == "?":
                    r[c] = mode_map[c]

    vocab = {}

    for c in cat_cols:

        s = set()

        for r in train_rows:
            s.add(r[c])

        vocab[c] = sorted(s)

    Xtr, Ytr, Ztr = encode_rows(train_rows, num_cols, cat_cols, vocab)
    Xte, Yte, Zte = encode_rows(test_rows, num_cols, cat_cols, vocab)

    Xtr, Xte = scale_data(Xtr, Xte)

    np.savez(split_dir / "adult_train.npz", x=Xtr, y=Ytr, z=Ztr)
    np.savez(split_dir / "adult_test.npz", x=Xte, y=Yte, z=Zte)

    print("adult dataset saved")
    print("train shape", Xtr.shape, "test shape", Xte.shape)


def main():

    a,b,c,d = make_dirs()

    make_synth_files(a,c)

    make_adult_npz(b,d)


if __name__ == "__main__":
    main()