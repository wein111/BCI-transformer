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
