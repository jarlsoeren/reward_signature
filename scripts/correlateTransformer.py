from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from scipy.stats import spearmanr

class CorrelateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, keep_ratio=0.5):
        self.keep_ratio = keep_ratio
        self.selected_idx_ = None

    def fit(self, X, y):
        # compute spearman correlation for each feature vs y
        corrs = []
        for i in range(X.shape[1]):
            if np.all(X[:, i] == X[0, i]):
                corrs.append(0)  # treat constant features as uninformative
                continue
            rho, _ = spearmanr(X[:, i], y)
            if np.isnan(rho):   # occurs when X[:, i] is constant
                rho = 0         # treat constant features as uninformative
            corrs.append(abs(rho))
        corrs = np.array(corrs)

        # keep the top fraction
        k = max(1, int(len(corrs) * self.keep_ratio))
        self.selected_idx_ = np.argsort(corrs)[-k:]
        return self

    def transform(self, X):
        return X[:, self.selected_idx_]
