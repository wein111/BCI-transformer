# preprocess_paper.py

import numpy as np
from scipy.ndimage import gaussian_filter1d

class PaperPreprocessor:

    def __init__(self, smooth_sigma=None):
        self.sigma = smooth_sigma

    def normalize(self, X, mean, std):
        return (X - mean) / (std + 1e-6)

    def smooth(self, X):
        if self.sigma is None:
            return X
        return gaussian_filter1d(X, sigma=self.sigma, axis=0)

    def __call__(self, X, mean, std):
        # 1. normalize
        X = self.normalize(X, mean, std)

        # 2. smooth (optional)
        if self.sigma is not None:
            X = self.smooth(X)

        # 3. 论文的 noise augment 在 train loop 中做，所以这里不做
        return X.astype(np.float32), X.shape[0]
