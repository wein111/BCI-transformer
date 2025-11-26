# train_original.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import editdistance
from model_GRU import LightGRUDecoder

from preprocess_original import OriginalPreprocessor
from ID2phoneme import id2phoneme


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# Dataset
# =========================================================
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

        # compute global mean/std
        self._compute_stats()

    # -------------------------------
    def _compute_stats(self):
        allX = []
        for ex in self.samples:
            e = tf.io.parse_single_example(ex, self.feature_desc)
            allX.append(e["inputFeatures"].numpy())

        Xcat = np.concatenate(allX, axis=0)
        self.mean = Xcat.mean(axis=0)
        self.std  = Xcat.std(axis=0) + 1e-5

    # -------------------------------
    def __len__(self):
        return len(self.samples)

    # -------------------------------
    def __getitem__(self, idx):
        e = tf.io.parse_single_example(self.samples[idx], self.feature_desc)

        X = e["inputFeatures"].numpy()
        y = e["seqClassIDs"].numpy()
        L = int(e["nSeqElements"].numpy())
        T = int(e["nTimeSteps"].numpy())

        y = y[:L]

        return X.astype(np.float32), y, T, L, self.mean, self.std


# =========================================================
# Collate
# =========================================================
def collate_fn(batch):
    Xs, ys, Ts, Ls, means, stds = zip(*batch)
    B = len(batch)
    T_max = max(Ts)

    X_pad = np.zeros((B, T_max, 256), dtype=np.float32)
    for i in range(B):
        X_pad[i, :Ts[i]] = Xs[i]

    # concat labels
    ycat = []
    for y in ys:
        ycat.extend(y.tolist())
    ycat = torch.tensor(ycat, dtype=torch.long)

    return (
        torch.tensor(X_pad, dtype=torch.float32),
        ycat,
        torch.tensor(Ls, dtype=torch.long),
        torch.tensor(Ts, dtype=torch.long),
        np.stack(means),
        np.stack(stds),
    )


# =========================================================
# Model: 6-layer GRU (bidirectional)
# =========================================================
class GRU6(nn.Module):
    def __init__(self, input_dim, hidden, num_layers, num_classes):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden,
            num_layers=num_layers,
            dropout=0.1,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        logits = self.fc(out)
        return logits.permute(1, 0, 2)   # (T,B,C)


# =========================================================
#  Greedy decode
# =========================================================
def greedy_decode(logits):
    pred = logits.argmax(dim=-1).cpu().tolist()

    seq = []
    prev = -1
    for p in pred:
        if p != 0 and p != prev:
            seq.append(p)
        prev = p
    return seq


# =========================================================
# Evaluate — you要求的
# =========================================================
def evaluate(model, loader, preproc):
    model.eval()

    total_pref = 0
    total_edit = 0
    total_cnt  = 0

    for (X, ycat, Ls, Ts, means, stds) in loader:

        X = X.numpy()
        B = X.shape[0]

        # preprocess
        X_new, T_new = [], []
        for i in range(B):
            Xi = X[i, :Ts[i]]
            Xi, Tnew = preproc(Xi, means[i], stds[i])
            X_new.append(Xi)
            T_new.append(Tnew)

        Tmax = max(T_new)
        feat_dim = X_new[0].shape[1]
        X_pad = np.zeros((B, Tmax, feat_dim), dtype=np.float32)
        for i in range(B):
            X_pad[i, :T_new[i]] = X_new[i]

        X_pad = torch.tensor(X_pad, dtype=torch.float32).to(DEVICE)
        T_new = torch.tensor(T_new, dtype=torch.long).to(DEVICE)
        ycat = ycat.to(DEVICE)
        Ls = Ls.to(DEVICE)

        # forward
        with torch.no_grad():
            logits = model(X_pad, T_new)
            log0 = logits[:, 0, :]

        pred = greedy_decode(log0)
        true = ycat[:Ls[0]].tolist()

        # prefix
        Lmin = min(len(pred), len(true))
        pref = sum(pred[i] == true[i] for i in range(Lmin)) / Lmin

        # edit
        edit = editdistance.eval(pred, true) / len(true)

        total_pref += pref
        total_edit += edit
        total_cnt  += 1

    return total_pref / total_cnt, total_edit / total_cnt


# =========================================================
# show_one 每个 epoch 看一条
# =========================================================
def show_one(model, loader, preproc):

    X, ycat, Ls, Ts, means, stds = next(iter(loader))
    X = X.numpy()

    Xi = X[0, :Ts[0]]
    Xi, Tnew = preproc(Xi, means[0], stds[0])

    Xi = torch.tensor(Xi[None, :, :], dtype=torch.float32).to(DEVICE)
    Tnew = torch.tensor([Tnew], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logits = model(Xi, Tnew)
        log0 = logits[:, 0, :]

    pred = greedy_decode(log0)
    true = ycat[:Ls[0]].tolist()

    print("\n===== Sample Decode =====")
    print("Pred IDs:", pred)
    print("True IDs:", true)
    print("Pred PH:", [id2phoneme[p] for p in pred])
    print("True PH:", [id2phoneme[p] for p in true])
    print("=========================\n")


# =========================================================
#  Train Loop
# =========================================================
def train():

    #BASE = "/Users/wei/Courses/EE675/SpeechBCI-Transformer/BCI-transformer/Dataset/derived/tfRecords"
    BASE = "/content/drive/MyDrive/EE675/BCI-transformer/Dataset/derived/tfRecords"
    DATES = [
        "t12.2022.04.28", "t12.2022.05.05", "t12.2022.05.17",
        "t12.2022.05.19", "t12.2022.05.24", "t12.2022.05.26",
        "t12.2022.06.02", "t12.2022.06.07", "t12.2022.06.14",
        "t12.2022.06.16", "t12.2022.06.21", "t12.2022.06.23",
        "t12.2022.06.28", "t12.2022.07.05", "t12.2022.07.14",
        "t12.2022.07.21", "t12.2022.07.27", "t12.2022.07.29"
    ]

    train_paths = [f"{BASE}/{d}/train/chunk_0.tfrecord" for d in DATES]
    test_paths  = [f"{BASE}/{d}/test/chunk_0.tfrecord"  for d in DATES]

    train_ds = SpikeDataset(train_paths)
    test_ds  = SpikeDataset(test_paths)

    train_loader = DataLoader(train_ds, batch_size=6, shuffle=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=6, shuffle=False, collate_fn=collate_fn)

    preproc = OriginalPreprocessor(
        smooth_sigma=1.5,
        stack_k=5,
        stack_stride=2,
        subsample_factor=3,
    )

    #input_dim = 256 * 7

    model = GRU6(
        input_dim=256 * 5,
        hidden=256,
        num_layers=6,
        num_classes=41
    ).to(DEVICE)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    from torch.optim import AdamW
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.90
    )

    best_edit = 999

    NUM_EPOCHS = 60

    for epoch in range(NUM_EPOCHS):

        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for X, ycat, Ls, Ts, means, stds in pbar:

            X = X.numpy()
            B = X.shape[0]

            # preprocess
            X_new, T_new = [], []
            for i in range(B):
                Xi = X[i, :Ts[i]]
                Xi, Tt = preproc(Xi, means[i], stds[i])
                X_new.append(Xi)
                T_new.append(Tt)

            Tmax = max(T_new)
            feat_dim = X_new[0].shape[1]

            X_pad = np.zeros((B, Tmax, feat_dim), dtype=np.float32)
            for i in range(B):
                X_pad[i, :T_new[i]] = X_new[i]

            X_pad = torch.tensor(X_pad, dtype=torch.float32).to(DEVICE)
            T_new = torch.tensor(T_new, dtype=torch.long).to(DEVICE)

            ycat = ycat.to(DEVICE)
            Ls   = Ls.to(DEVICE)

            logits = model(X_pad, T_new)
            log_probs = logits.log_softmax(dim=-1)

            loss = criterion(log_probs, ycat, T_new.cpu(), Ls.cpu())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        scheduler.step()

        # 评估
        train_pref, train_edit = evaluate(model, train_loader, preproc)
        test_pref,  test_edit  = evaluate(model, test_loader,  preproc)

        print(f"\n[Epoch {epoch}] Loss={total_loss:.1f}")
        print(f" Train: pref={train_pref:.3f} | PER={train_edit:.3f}")
        print(f" Test : pref={test_pref:.3f}  | PER={test_edit:.3f}")

        show_one(model, test_loader, preproc)

        # save best
        if test_edit < best_edit:
            best_edit = test_edit
            torch.save(model.state_dict(), "best_model.pt")
            print(">>> Saved BEST model\n")


if __name__ == "__main__":
    train()
