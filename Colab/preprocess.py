# preprocess_torch.py
import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter1d

class SpikePreprocessor:

    def __init__(self,
                 smooth_sigma=2,
                 stack_k=5,
                 stack_stride=2,
                 subsample_factor=2):

        self.smooth_sigma = smooth_sigma
        self.stack_k = stack_k
        self.stack_stride = stack_stride
        self.subsample_factor = subsample_factor

    # ---------------- Smooth (per sample) ----------------
    def smooth_single(self, x):
        # x: (T,C) numpy array
        return gaussian_filter1d(x, sigma=self.smooth_sigma, axis=0)

    # ---------------- Time stack ----------------
    def stack_single(self, X):
        T, C = X.shape
        k, s = self.stack_k, self.stack_stride

        T_new = (T - k) // s + 1
        out = np.zeros((T_new, C * k), dtype=np.float32)

        for i, t in enumerate(range(0, T - k + 1, s)):
            out[i] = X[t:t+k].reshape(-1)

        return out

    # ---------------- Subsample ----------------
    def subsample_single(self, X):
        return X[::self.subsample_factor]

    # ---------------- Main batch API ----------------
    def preprocess(self, X, Ts):
        """
        X:  (B, T_max, C) torch tensor
        Ts: (B) lengths
        return: X_pp (B, T_pp_max, C'), Ts_new
        """
        X = X.cpu().numpy()
        Ts = Ts.cpu().numpy()

        X_pp_list = []
        Ts_new = []

        for i in range(len(X)):
            x = X[i, :Ts[i]]     # (T_i, C)
            x = self.smooth_single(x)
            x = self.stack_single(x)
            x = self.subsample_single(x)
            X_pp_list.append(x)
            Ts_new.append(x.shape[0])

        # pad to B, T_max_new, C_new
        T_max_new = max(Ts_new)
        C_new = X_pp_list[0].shape[1]

        X_pad = np.zeros((len(X), T_max_new, C_new), dtype=np.float32)
        for i, xp in enumerate(X_pp_list):
            X_pad[i, :xp.shape[0]] = xp

        return torch.tensor(X_pad, dtype=torch.float32), torch.tensor(Ts_new, dtype=torch.long)
