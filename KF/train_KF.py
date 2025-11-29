# ===========================================================
# train_KF.py
# Kalman Filter + Logistic Regression baseline (one-shot)
# ===========================================================

import torch
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from preprocess_kf import KalmanPreprocessor
from ID2phoneme import id2phoneme
import editdistance

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===========================================================
#  Dataset
# ===========================================================
class SpikeDataset(Dataset):
    def __init__(self, tfrecord_paths, max_len=500):
        self.paths = tfrecord_paths
        self.max_len = max_len

        self.raw = tf.data.TFRecordDataset(tfrecord_paths)

        self.feature_desc = {
            "inputFeatures": tf.io.FixedLenSequenceFeature([256], tf.float32, allow_missing=True),
            "seqClassIDs": tf.io.FixedLenFeature((max_len,), tf.int64),
            "nSeqElements": tf.io.FixedLenFeature((), tf.int64),
            "nTimeSteps": tf.io.FixedLenFeature((), tf.int64),
        }

        self.samples = list(self.raw)
        self._compute_stats()

    def _compute_stats(self):
        allX = []
        for ex in self.samples:
            e = tf.io.parse_single_example(ex, self.feature_desc)
            allX.append(e["inputFeatures"].numpy())

        Xcat = np.concatenate(allX, axis=0)
        self.mean = Xcat.mean(axis=0)
        self.std = Xcat.std(axis=0) + 1e-5

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        e = tf.io.parse_single_example(self.samples[idx], self.feature_desc)

        X = e["inputFeatures"].numpy().astype(np.float32)
        y = e["seqClassIDs"].numpy().reshape(-1).astype(int)
        L = int(e["nSeqElements"].numpy())
        T = int(e["nTimeSteps"].numpy())

        y = y[:L]
        return X, y, T, L, self.mean, self.std


# ===========================================================
# 1. Fit LDS (A, C, Q, R) - One-shot estimation
# ===========================================================
def fit_lds(Xs, latent_dim=30):
    """Fit LDS parameters using PCA + least squares (one-shot)."""
    allX = np.concatenate(Xs, axis=0)

    print("Running PCA for initialization...")
    allX = allX - allX.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(allX, full_matrices=False)
    C = Vt[:latent_dim].T      # (256, K)
    pinvC = np.linalg.pinv(C)  # (K, 256)

    # Project to latent
    X_latents = [X @ pinvC.T for X in Xs]

    print("Estimating A...")
    K = latent_dim
    A_num = np.zeros((K, K))
    A_den = np.zeros((K, K))

    for Z in X_latents:
        if len(Z) < 2:
            continue
        Zt = Z[:-1]
        Zt1 = Z[1:]
        A_num += Zt1.T @ Zt
        A_den += Zt.T @ Zt

    A = A_num @ np.linalg.pinv(A_den)

    print("Estimating Q...")
    Qs = []
    for Z in X_latents:
        if len(Z) < 2:
            continue
        err = Z[1:] - Z[:-1] @ A.T
        Qs.append(err.T @ err / len(err))
    Q = np.mean(Qs, axis=0)

    print("Estimating R...")
    Rs = []
    for X, Z in zip(Xs, X_latents):
        err = X - Z @ C.T
        Rs.append(err.T @ err / len(err))
    R = np.mean(Rs, axis=0)

    return A, C, Q, R, X_latents


# ===========================================================
# 2. Kalman Filter forward
# ===========================================================
def kalman_filter(Y, A, C, Q, R):
    T, D = Y.shape
    K = A.shape[0]

    x = np.zeros(K)
    P = np.eye(K)

    out = np.zeros((T, K))

    for t in range(T):
        # Predict
        x_pred = A @ x
        P_pred = A @ P @ A.T + Q

        # Update
        y = Y[t]
        S = C @ P_pred @ C.T + R
        K_gain = P_pred @ C.T @ np.linalg.pinv(S)

        x = x_pred + K_gain @ (y - C @ x_pred)
        P = (np.eye(K) - K_gain @ C) @ P_pred

        out[t] = x

    return out


# ===========================================================
# 3. Create frame-level labels (uniform alignment)
# ===========================================================
def create_frame_labels(T_frames, phoneme_seq):
    """Uniformly align phoneme sequence to frame sequence."""
    L = len(phoneme_seq)
    if L == 0:
        return np.zeros(T_frames, dtype=int)
    
    frame_labels = np.zeros(T_frames, dtype=int)
    frames_per_phoneme = T_frames / L
    
    for i in range(T_frames):
        phoneme_idx = min(int(i / frames_per_phoneme), L - 1)
        frame_labels[i] = phoneme_seq[phoneme_idx]
    
    return frame_labels


# ===========================================================
# 4. Train classifier (Logistic Regression)
# ===========================================================
from sklearn.linear_model import LogisticRegression

def train_classifier(latent_list, label_list):
    """Train classifier with proper frame-level alignment."""
    X_all = []
    y_all = []

    for Z, y in zip(latent_list, label_list):
        T_frames = len(Z)
        y = np.asarray(y).reshape(-1).astype(int)
        frame_labels = create_frame_labels(T_frames, y)
        X_all.append(Z)
        y_all.append(frame_labels)

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    
    print(f"Training classifier on {len(X_all)} frames...")
    print(f"Label distribution: {np.bincount(y_all)}")
    
    clf = LogisticRegression(max_iter=500, class_weight='balanced', n_jobs=-1)
    clf.fit(X_all, y_all)
    return clf


# ===========================================================
# 5. Decode sequence with smoothing
# ===========================================================
from scipy.stats import mode

def smooth_predictions(pred, window_size=5):
    """Apply majority voting smoothing."""
    T = len(pred)
    smoothed = np.zeros(T, dtype=int)
    half_w = window_size // 2
    
    for i in range(T):
        start = max(0, i - half_w)
        end = min(T, i + half_w + 1)
        window = pred[start:end]
        smoothed[i] = mode(window, keepdims=False)[0]
    
    return smoothed


def remove_short_segments(pred, min_len=3):
    """Remove segments shorter than min_len frames."""
    result = pred.copy()
    T = len(pred)
    
    i = 0
    while i < T:
        j = i
        while j < T and pred[j] == pred[i]:
            j += 1
        seg_len = j - i
        
        if seg_len < min_len:
            if i > 0:
                result[i:j] = result[i-1]
            elif j < T:
                result[i:j] = pred[j]
        
        i = j
    
    return result


def decode_sequence(Y, A, C, Q, R, clf, smooth=True):
    """Decode with optional smoothing."""
    Z = kalman_filter(Y, A, C, Q, R)
    pred = clf.predict(Z)
    
    if smooth:
        pred = smooth_predictions(pred, window_size=7)
        pred = remove_short_segments(pred, min_len=5)
    
    return pred


def collapse_repeats(seq):
    """Collapse consecutive repeated phonemes."""
    if len(seq) == 0:
        return []
    result = [seq[0]]
    for i in range(1, len(seq)):
        if seq[i] != result[-1]:
            result.append(seq[i])
    return result


# ===========================================================
# 6. Evaluation
# ===========================================================
def evaluate_kf(test_loader, preproc, A, C, Q, R, clf, num_samples=20):
    total_pref_frame = 0
    total_edit_seq = 0
    count = 0

    print(f"Evaluating {num_samples} random samples...")

    for i, (X, y, T, L, mean, std) in enumerate(test_loader):
        if count >= num_samples:
            break

        Xi = X[0, :T].numpy()
        yi = y.numpy().reshape(-1).astype(int)
        L_val = int(L.numpy())
        yi = yi[:L_val]

        Xi, T_proc = preproc(Xi, mean.numpy(), std.numpy())
        pred_frames = decode_sequence(Xi, A, C, Q, R, clf)
        
        yi_frames = create_frame_labels(len(pred_frames), yi)
        pref_frame = np.mean(pred_frames == yi_frames)
        
        pred_seq = collapse_repeats(pred_frames.tolist())
        edit_seq = editdistance.eval(pred_seq, yi.tolist()) / max(len(yi), 1)

        total_pref_frame += pref_frame
        total_edit_seq += edit_seq
        count += 1

    return total_pref_frame / count, total_edit_seq / count


def show_one(test_loader, preproc, A, C, Q, R, clf):
    X, y, T, L, mean, std = next(iter(test_loader))

    Xi = X[0, :T].numpy()
    yi = y.numpy().reshape(-1).astype(int)
    L_val = int(L.numpy())
    yi = yi[:L_val]

    Xi, _ = preproc(Xi, mean.numpy(), std.numpy())
    pred_frames = decode_sequence(Xi, A, C, Q, R, clf)
    pred_seq = collapse_repeats(pred_frames.tolist())

    print("\n===== KF Sample Decode =====")
    print(f"Frame predictions (first 50): {pred_frames[:50].tolist()}")
    print(f"Pred Seq IDs ({len(pred_seq)}): {pred_seq[:30]}...")
    print(f"True Seq IDs ({len(yi)}): {yi.tolist()[:30]}...")
    print(f"Pred PH: {[id2phoneme.get(p, '?') for p in pred_seq[:20]]}...")
    print(f"True PH: {[id2phoneme.get(p, '?') for p in yi[:20]]}...")
    print(f"Edit distance: {editdistance.eval(pred_seq, yi.tolist())}")
    print("================================\n")


# ===========================================================
# 7. Main training entry
# ===========================================================
def train_kf():
    BASE = r"D:\DeepLearning\BCI\Dataset\derived\tfRecords"
    DATES = [
        "t12.2022.04.28", "t12.2022.05.05", "t12.2022.05.17",
        "t12.2022.05.19", "t12.2022.05.24", "t12.2022.05.26",
        "t12.2022.06.02", "t12.2022.06.07", "t12.2022.06.14",
        "t12.2022.06.16", "t12.2022.06.21", "t12.2022.06.23",
        "t12.2022.06.28", "t12.2022.07.05", "t12.2022.07.14",
        "t12.2022.07.21", "t12.2022.07.27", "t12.2022.07.29"
    ]

    train_paths = [f"{BASE}/{d}/train/chunk_0.tfrecord" for d in DATES]
    test_paths = [f"{BASE}/{d}/test/chunk_0.tfrecord" for d in DATES]

    train_ds = SpikeDataset(train_paths)
    test_ds = SpikeDataset(test_paths)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=True)

    preproc = KalmanPreprocessor(sigma=1.5)

    # =======================
    # Load + preprocess train
    # =======================
    print("Loading & preprocessing training data...")
    Xs = []
    Ys = []

    for X, y, T, L, mean, std in tqdm(train_loader):
        Xi = X[0, :T].numpy()
        yi = y.numpy().reshape(-1).astype(int)

        Xi, _ = preproc(Xi, mean.numpy(), std.numpy())
        Xs.append(Xi)
        Ys.append(yi)

    # =======================
    # Fit LDS (one-shot)
    # =======================
    print("\nFitting LDS (one-shot)...")
    A, C, Q, R, latents = fit_lds(Xs, latent_dim=30)

    # =======================
    # Fit classifier
    # =======================
    print("\nTraining classifier...")
    clf = train_classifier(latents, Ys)

    # =======================
    # Evaluate
    # =======================
    frame_acc, seq_per = evaluate_kf(test_loader, preproc, A, C, Q, R, clf, num_samples=20)
    print(f"\nKF Results:")
    print(f"  Frame-level Accuracy: {frame_acc:.3f}")
    print(f"  Sequence PER (after collapse): {seq_per:.3f}")

    # =======================
    # Show one sample
    # =======================
    show_one(test_loader, preproc, A, C, Q, R, clf)


if __name__ == "__main__":
    train_kf()

