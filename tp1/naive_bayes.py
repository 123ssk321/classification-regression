import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import KernelDensity
from sklearn.metrics import accuracy_score


class NaiveBayes(BaseEstimator):
    def __init__(self, bandwidth, classes):
        self.bandwidth = bandwidth
        self.classes = classes
        self.kdes = []
        self.log_priors = []

    def fit(self, X, y):
        if len(self.kdes) != 0:
            self.kdes = []

        self.log_priors = [np.log(y[y == cls].shape[0] / y.shape[0]) for cls in self.classes]
        for cls in self.classes:
            X_cls, y_cls = self._get_class_rows(X, y, cls)

            kdes_cls = []
            for feat_col in range(X_cls.shape[1]):
                col_arr = X_cls[:, feat_col].reshape(-1, 1)
                kde_cls = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth).fit(col_arr)
                kdes_cls.append(kde_cls)
            self.kdes.append(kdes_cls)
        return self

    def predict(self, X):
        preds = []
        for row in X:
            totals = []
            for cls in self.classes:
                totals.append(self.log_priors[cls] + np.sum([kde.score_samples(x.reshape(1, 1))
                                                             for kde, x in zip(self.kdes[cls], row)]))
            preds.append(np.argmax(totals, axis=0))

        return preds

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def _get_class_rows(self, X, y, cls):
        class_rows = y == cls
        return X[class_rows], y[class_rows]
