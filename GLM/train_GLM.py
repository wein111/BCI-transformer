# ===========================================================
# train_GLM.py
# Generalized Linear Model (GLM) baseline for BCI decoding
# GLM = Multinomial Logistic Regression (softmax regression)
# ===========================================================

import torch
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from preprocess_glm import GLMPreprocessor
from ID2phoneme import id2phoneme
import editdistance


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
# 1. Create frame-level labels (uniform alignment)
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
# 2. Add temporal context features
# ===========================================================
def add_temporal_context(X, context_size=2):
    """
    Add temporal context by stacking neighboring frames.
    
    Args:
        X: (T, D) feature matrix
        context_size: number of frames before/after to include
    
    Returns:
        X_context: (T, D * (2*context_size + 1)) with temporal context
    """
    T, D = X.shape
    X_padded = np.pad(X, ((context_size, context_size), (0, 0)), mode='edge')
    
    X_context = np.zeros((T, D * (2 * context_size + 1)))
    for i in range(T):
        X_context[i] = X_padded[i:i + 2 * context_size + 1].flatten()
    
    return X_context


# ===========================================================
# 3. Train GLM (Multinomial Logistic Regression)
# ===========================================================
from sklearn.linear_model import LogisticRegression

def train_glm(X_list, Y_list, use_temporal=True, context_size=2):
    """
    Train GLM (multinomial logistic regression) with frame-level alignment.
    
    GLM: P(y|x) = softmax(W @ x + b)
    
    Args:
        use_temporal: Add temporal context features
        context_size: Number of frames before/after to include
    """
    X_all = []
    y_all = []

    for X, y in zip(X_list, Y_list):
        T_frames = len(X)
        y = np.asarray(y).reshape(-1).astype(int)
        frame_labels = create_frame_labels(T_frames, y)
        
        # Add temporal context
        if use_temporal:
            X = add_temporal_context(X, context_size=context_size)
        
        X_all.append(X)
        y_all.append(frame_labels)

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    
    print(f"Training GLM on {len(X_all)} frames...")
    print(f"Feature dim: {X_all.shape[1]}, Classes: {len(np.unique(y_all))}")
    if use_temporal:
        print(f"Using temporal context: {context_size} frames before/after")
    
    # GLM = Multinomial Logistic Regression
    glm = LogisticRegression(
        solver='lbfgs',
        max_iter=500,
        class_weight='balanced',
        n_jobs=-1,
        verbose=1
    )
    glm.fit(X_all, y_all)
    
    return glm, use_temporal, context_size


# ===========================================================
# 3. Decode with smoothing
# ===========================================================
from scipy.stats import mode

def smooth_predictions(pred, window_size=7):
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


def remove_short_segments(pred, min_len=5):
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


def decode_sequence(X, glm, smooth=True, use_temporal=False, context_size=2):
    """Decode sequence with GLM."""
    if use_temporal:
        X = add_temporal_context(X, context_size=context_size)
    
    pred = glm.predict(X)
    
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
# 5. Evaluation
# ===========================================================
def evaluate_glm(test_loader, preproc, glm, num_samples=20, use_temporal=False, context_size=2):
    total_pref_frame = 0
    total_edit_seq = 0
    count = 0

    print(f"Evaluating {num_samples} samples...")

    for i, (X, y, T, L, mean, std) in enumerate(test_loader):
        if count >= num_samples:
            break

        Xi = X[0, :T].numpy()
        yi = y.numpy().reshape(-1).astype(int)
        L_val = int(L.numpy())
        yi = yi[:L_val]

        Xi, _ = preproc(Xi, mean.numpy(), std.numpy())
        pred_frames = decode_sequence(Xi, glm, use_temporal=use_temporal, context_size=context_size)
        
        yi_frames = create_frame_labels(len(pred_frames), yi)
        pref_frame = np.mean(pred_frames == yi_frames)
        
        pred_seq = collapse_repeats(pred_frames.tolist())
        edit_seq = editdistance.eval(pred_seq, yi.tolist()) / max(len(yi), 1)

        total_pref_frame += pref_frame
        total_edit_seq += edit_seq
        count += 1

    return total_pref_frame / count, total_edit_seq / count


def show_one(test_loader, preproc, glm, use_temporal=False, context_size=2):
    X, y, T, L, mean, std = next(iter(test_loader))

    Xi = X[0, :T].numpy()
    yi = y.numpy().reshape(-1).astype(int)
    L_val = int(L.numpy())
    yi = yi[:L_val]

    Xi, _ = preproc(Xi, mean.numpy(), std.numpy())
    pred_frames = decode_sequence(Xi, glm, use_temporal=use_temporal, context_size=context_size)
    pred_seq = collapse_repeats(pred_frames.tolist())

    print("\n===== GLM Sample Decode =====")
    print(f"Frame predictions (first 50): {pred_frames[:50].tolist()}")
    print(f"Pred Seq IDs ({len(pred_seq)}): {pred_seq[:30]}...")
    print(f"True Seq IDs ({len(yi)}): {yi.tolist()[:30]}...")
    print(f"Pred PH: {[id2phoneme.get(p, '?') for p in pred_seq[:20]]}...")
    print(f"True PH: {[id2phoneme.get(p, '?') for p in yi[:20]]}...")
    print(f"Edit distance: {editdistance.eval(pred_seq, yi.tolist())}")
    print("==============================\n")


# ===========================================================
# 6. Main training entry
# ===========================================================
def train_glm_main():
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
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)  # No shuffle for stable eval

    preproc = GLMPreprocessor(sigma=1.5)

    # =======================
    # Config
    # =======================
    USE_TEMPORAL = False  # Disabled: 1280维太大，内存不足
    CONTEXT_SIZE = 2      # ±2 frames = 5 frames total (if enabled)

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
    # Train GLM
    # =======================
    print("\nTraining GLM (Multinomial Logistic Regression)...")
    glm, use_temporal, context_size = train_glm(Xs, Ys, use_temporal=USE_TEMPORAL, context_size=CONTEXT_SIZE)

    # =======================
    # Evaluate (more samples for stability)
    # =======================
    frame_acc, seq_per = evaluate_glm(test_loader, preproc, glm, num_samples=50, 
                                       use_temporal=use_temporal, context_size=context_size)
    print(f"\nGLM Results:")
    print(f"  Frame-level Accuracy: {frame_acc:.3f}")
    print(f"  Sequence PER (after collapse): {seq_per:.3f}")

    # =======================
    # Show one sample
    # =======================
    show_one(test_loader, preproc, glm, use_temporal=use_temporal, context_size=context_size)


if __name__ == "__main__":
    train_glm_main()

