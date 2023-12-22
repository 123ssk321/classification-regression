import math
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tqdm import tqdm
from naive_bayes import NaiveBayes


def naive_bayes_classifier(X_train, y_train, X_val, y_val, bandwidth, classes):
    nb = NaiveBayes(bandwidth, classes)
    train_err = 1 - nb.fit(X_train, y_train).score(X_train, y_train)

    val_err = 1 - nb.score(X_val, y_val)
    return train_err, val_err


def cv_naive_bayes(X_train, y_train, X_scaler, classes=None, folds=5):
    if classes is None:
        classes = [0, 1]
    skf = StratifiedKFold(n_splits=folds)

    avg_train_errs = []
    avg_val_errs = []
    bandwidths = np.arange(0.02, 0.62, 0.02)
    for bandwidth in tqdm(bandwidths):

        sum_train_err = 0
        sum_val_err = 0
        for train, valid in skf.split(X_train, y_train):
            X_r = X_train.iloc[train]
            y_r = y_train.iloc[train].to_numpy()

            X_v = X_train.iloc[valid]
            y_v = y_train.iloc[valid].to_numpy()

            X_r = X_scaler.fit_transform(X_r)
            X_v = X_scaler.transform(X_v)

            train_err, val_err = naive_bayes_classifier(X_r, y_r, X_v, y_v, bandwidth, classes)

            sum_train_err += train_err
            sum_val_err += val_err

        avg_train_errs.append(sum_train_err / folds)
        avg_val_errs.append(sum_val_err / folds)

    best_model_ix = np.argmin(avg_val_errs, axis=0)
    best_bandwidth = bandwidths[best_model_ix]
    best_model_train_err = avg_train_errs[best_model_ix]
    best_model_val_err = avg_val_errs[best_model_ix]

    plot_err(avg_train_errs, avg_val_errs)
    return best_bandwidth, best_model_train_err, best_model_val_err


def gaussian_naive_bayes(X_train, y_train):
    clf = GaussianNB()
    clf.fit(X_train, y_train)

    train_err = 1 - clf.score(X_train, y_train)
    return clf, train_err


def plot_err(train_errs, val_errs):
    bandwidths = np.arange(0.02, 0.62, 0.02)
    plt.plot(bandwidths, train_errs, label='train')
    plt.plot(bandwidths, val_errs, label='val')
    plt.xlabel('bandwidth')
    plt.ylabel('error')

    plt.legend()
    plt.show()


def approximate_normal_test_95(nb_err, gnb_err, test_size):
    nb_std_dev = math.sqrt(test_size * nb_err*(1 - nb_err))
    gnb_std_dev = math.sqrt(test_size * gnb_err*(1 - gnb_err))

    nb_n_err = nb_err * test_size
    gnb_n_err = gnb_err * test_size

    return (nb_n_err, 1.96*nb_std_dev), (gnb_n_err, 1.96*gnb_std_dev)


def mcNemar_test_95(nb, gnb, X_test, y_test):
    X_test_arr = X_test.to_numpy()
    y_test_arr = y_test.to_numpy()

    nb_preds = nb.predict(X_test_arr)
    gnb_preds = gnb.predict(X_test_arr)

    e01 = e10 = 0
    for nb_pred, gnb_pred, y_true in zip(nb_preds, gnb_preds, y_test_arr):
        if nb_pred != y_true and gnb_pred == y_true:
            e01 += 1
        if gnb_pred != y_true and nb_pred == y_true:
            e10 += 1

    chi = ((abs(e01-e10) - 1)**2)/(e01+e10)
    return chi


def fake_bank_note_detection(train_set_path='../data/TP1_train.tsv', test_set_path='../data/TP1_test.tsv'):
    train_dataset = pd.read_csv(train_set_path, sep='\t', header=None)
    train_dataset = shuffle(train_dataset)

    test_dataset = pd.read_csv(test_set_path, sep='\t', header=None)
    test_dataset = shuffle(test_dataset)

    X_train = train_dataset[train_dataset.columns.values.tolist()[:len(train_dataset.columns) - 1]]
    y_train = train_dataset[train_dataset.columns.values.tolist()[-1]]

    X_test = test_dataset[test_dataset.columns.values.tolist()[:len(test_dataset.columns) - 1]]
    y_test = test_dataset[test_dataset.columns.values.tolist()[-1]]

    X_scaler = StandardScaler()

    bandwidth, nb_train_err, nb_val_err = cv_naive_bayes(X_train, y_train, X_scaler)

    X_r = X_scaler.fit_transform(X_train)
    X_t = X_scaler.transform(X_test)

    nb = NaiveBayes(bandwidth, classes=[0, 1])
    nb_test_err = 1 - nb.fit(X_r, y_train.to_numpy()).score(X_t, y_test.to_numpy())
    print(f'Best Naive Bayes model bandwidth: {bandwidth:.2f}\t'
          f'Train error: {nb_train_err:.2f}\tValidation error: {nb_val_err:.2f}\tTest error: {nb_test_err:.2f}')

    gnb, gnb_train_err = gaussian_naive_bayes(X_r, y_train)

    gnb_test_err = 1 - gnb.score(X_t, y_test.to_numpy())
    print(f'Gaussian Naive Bayes model test error: {gnb_test_err:.2f}')

    print('Approximate normal test')
    nb_res, gnb_res = approximate_normal_test_95(nb_test_err, gnb_test_err, len(X_test))
    print(f'Naive Bayes: {nb_res[0]:.2f} \u00B1 {nb_res[1]:.2f}\t'
          f'Gaussian Naive Bayes: {gnb_res[0]:.2f} \u00B1 {gnb_res[1]:.2f}')

    print('McNemar\'s test')
    chi = mcNemar_test_95(nb, gnb, X_test, y_test)
    print(f'Naive Bayes vs Gaussian Naive Bayes: {chi}')
