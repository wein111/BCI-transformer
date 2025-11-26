import tensorflow as tf
import numpy as np
from torch.utils.data import Dataset
from preprocess import SpikePreprocessor

class SpikeDataset(Dataset):
    def __init__(self,
                 tfrecord_paths,
                 max_target_len=500,
                 compute_norm=True,
                 white_noise=0.01,
                 static_gain=0.05,
                 smooth_sigma=2,
                 stack_k=5,
                 stack_stride=2,
                 subsample_factor=2):

        self.tfrecord_paths = tfrecord_paths
        self.white_noise = white_noise
        self.static_gain = static_gain
        self.max_target_len = max_target_len

        self.pre = SpikePreprocessor(
            smooth_sigma=smooth_sigma,
            stack_k=stack_k,
            stack_stride=stack_stride,
            subsample_factor=subsample_factor,
        )

        self.raw = tf.data.TFRecordDataset(tfrecord_paths)

        self.feature_description = {
            "inputFeatures": tf.io.FixedLenSequenceFeature([256], tf.float32, allow_missing=True),
            "seqClassIDs": tf.io.FixedLenFeature((max_target_len,), tf.int64),
            "nSeqElements": tf.io.FixedLenFeature((), tf.int64),
            "nTimeSteps": tf.io.FixedLenFeature((), tf.int64),
        }

        self.samples = list(self.raw)

        self.session_stats = {}
        if compute_norm:
            self._compute_session_stats()

    def _compute_session_stats(self):
        from collections import defaultdict
        tmp = defaultdict(list)

        for path, example in zip(self.tfrecord_paths, self.samples):
            e = tf.io.parse_single_example(example, self.feature_description)
            X = e["inputFeatures"].numpy()
            tmp[path].append(X)

        for path, arr_list in tmp.items():
            Xcat = np.concatenate(arr_list, axis=0)
            mean = Xcat.mean(axis=0)
            std = Xcat.std(axis=0) + 1e-5
            self.session_stats[path] = (mean, std)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.tfrecord_paths[idx % len(self.tfrecord_paths)]
        example = self.samples[idx]
        e = tf.io.parse_single_example(example, self.feature_description)

        X = e["inputFeatures"].numpy().astype(np.float32)
        y = e["seqClassIDs"].numpy()
        L = int(e["nSeqElements"].numpy())
        T = int(e["nTimeSteps"].numpy())
        y = y[:L]

        if path in self.session_stats:
            mean, std = self.session_stats[path]
            X = (X - mean) / std

        if self.white_noise > 0:
            X = X + np.random.randn(*X.shape) * self.white_noise

        if self.static_gain > 0:
            gain = 1.0 + np.random.randn() * self.static_gain
            X = X * gain



        return X, y, T, L


import torch
import numpy as np

def collate_fn(batch):
    """
    batch: list of (X, y, T, L)
      X: (T,256)
      y: (L,)
      T: int (orig time)
      L: int (label length)

    Output:
      X_pad: (B, T_max, 256)
      y_cat: (sum L,)
      Ls: (B,)
      Ts: (B,)
    """

    Xs = [b[0] for b in batch]
    ys = [b[1] for b in batch]
    Ts = [b[2] for b in batch]
    Ls = [b[3] for b in batch]

    B = len(batch)
    T_max = max(Ts)  # pad to same length

    # pad X
    X_pad = np.zeros((B, T_max, 256), dtype=np.float32)
    for i, X in enumerate(Xs):
        T = X.shape[0]
        X_pad[i, :T] = X

    # concatenate labels (CTC expected format)
    y_cat = np.concatenate(ys, axis=0)

    # convert to torch
    X_pad = torch.tensor(X_pad, dtype=torch.float32)
    y_cat = torch.tensor(y_cat, dtype=torch.long)
    Ls = torch.tensor(Ls, dtype=torch.long)  # label len
    Ts = torch.tensor(Ts, dtype=torch.long)  # original frame count

    return X_pad, y_cat, Ls, Ts
