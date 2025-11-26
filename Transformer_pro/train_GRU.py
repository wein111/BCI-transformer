# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import SpikeDataset
from preprocess import SpikePreprocessor
from model_GRU import LightGRUDecoder

import numpy as np
from editdistance import eval as edit_distance

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------
# collate_fn for padding variable-length sequences
# ------------------------------------------------------
def collate_fn(batch):
    """
    batch: list of (X, y, T, L)
    Output:
        X_pad: (B, T_max, C)
        y_cat: concatenated labels
        Ls: label lengths
        Ts: input lengths
    """
    Xs, ys, Ts, Ls = zip(*batch)
    B = len(batch)
    C = Xs[0].shape[1]

    T_max = max(Ts)
    X_pad = np.zeros((B, T_max, C), dtype=np.float32)

    y_cat = []
    for i, (X, y, T, L) in enumerate(batch):
        X_pad[i, :T, :] = X
        y_cat.extend(y.tolist()[:L])     # flatten to 1D list

    y_cat = torch.tensor(y_cat, dtype=torch.long)
    X_pad = torch.tensor(X_pad, dtype=torch.float32)
    Ls = torch.tensor(Ls, dtype=torch.long)
    Ts = torch.tensor(Ts, dtype=torch.long)

    return X_pad, y_cat, Ls, Ts


# ------------------------------------------------------
# Simple CTC greedy decoder (optional)
# ------------------------------------------------------
def ctc_greedy_decode(logits, blank=0):
    # logits: (T, C)
    pred = torch.argmax(logits, dim=-1).tolist()
    out = []
    prev = None
    for p in pred:
        if p != blank and p != prev:
            out.append(p)
        prev = p
    return out


# ------------------------------------------------------
# Edit accuracy (for CER)
# ------------------------------------------------------
def edit_accuracy(pred, true):
    return edit_distance(pred, true) / max(1, len(true))


# ------------------------------------------------------
# Evaluate (你的原版格式)
# ------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, preprocessor, max_batches=30, beam=False):
    model.eval()

    total_pref = 0.0
    total_edit = 0.0
    total_cnt = 0

    for bi, (X, y_cat, Ls, Ts) in enumerate(loader):
        if bi >= max_batches:
            break

        X = X.to(DEVICE)
        Ts = Ts.to(DEVICE)

        # ---- preprocess ----
        X_pp, Ts_new = preprocessor.preprocess(X, Ts)
        X_pp = X_pp.to(DEVICE)

        # ---- forward ----
        logits = model(X_pp, Ts_new)      # (T, B, C)
        logits = logits[:, 0]             # first sample

        # ---- target ----
        L = Ls[0].item()
        true = y_cat[:L].tolist()

        # ---- decode ----
        if beam:
            pred = ctc_greedy_decode(logits)   # 你也可换 beam
        else:
            pred = ctc_greedy_decode(logits)

        # ---- prefix acc ----
        Lp = min(len(pred), len(true))
        pref = sum(pred[i] == true[i] for i in range(Lp))/Lp if Lp > 0 else 0.0

        # ---- edit ----
        edit = edit_accuracy(pred, true)

        total_pref += pref
        total_edit += edit
        total_cnt += 1

    if total_cnt == 0:
        return 0, 0

    return total_pref/total_cnt, total_edit/total_cnt


# ------------------------------------------------------
# Show one sample
# ------------------------------------------------------
@torch.no_grad()
def show_one(model, loader, preprocessor, id2phoneme, title=""):
    X, y_cat, Ls, Ts = next(iter(loader))

    X = X.to(DEVICE)
    Ts = Ts.to(DEVICE)

    X_pp, Ts_new = preprocessor.preprocess(X, Ts)
    X_pp = X_pp.to(DEVICE)

    logits = model(X_pp, Ts_new)
    logits = logits[:, 0]

    L = Ls[0].item()
    true_ids = y_cat[:L].tolist()
    pred_ids = ctc_greedy_decode(logits)

    print(f"\n==== {title} Decode ====")
    print("Pred IDs:", pred_ids)
    print("True IDs:", true_ids)
    print("Pred ph:", [id2phoneme.get(i,"?") for i in pred_ids])
    print("True ph:", [id2phoneme.get(i,"?") for i in true_ids])
    print("=================================\n")


# ------------------------------------------------------
# TRAIN LOOP
# ------------------------------------------------------
def train():

    # ------------ Load dataset --------------
    from ID2phoneme import id2phoneme

    BASE = "/Users/wei/Courses/EE675/SpeechBCI-Transformer/BCI-transformer/Dataset/derived/tfRecords"
    TRAIN_DATES = [
        "t12.2022.04.28", "t12.2022.05.05", "t12.2022.05.17",
        "t12.2022.05.19", "t12.2022.05.24", "t12.2022.05.26",
        "t12.2022.06.02", "t12.2022.06.07", "t12.2022.06.14",
        "t12.2022.06.16", "t12.2022.06.21", "t12.2022.06.23"
    ]
    TEST_DATES = TRAIN_DATES

    train_paths = [f"{BASE}/{d}/train/chunk_0.tfrecord" for d in TRAIN_DATES]
    test_paths  = [f"{BASE}/{d}/test/chunk_0.tfrecord" for d in TEST_DATES]

    train_dataset = SpikeDataset(train_paths)
    test_dataset  = SpikeDataset(test_paths)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # ------------ Preprocessor --------------
    preproc = SpikePreprocessor(
        smooth_sigma=2,
        stack_k=5,
        stack_stride=2,
        subsample_factor=2
    )



    # ------------ Model --------------
    model = LightGRUDecoder(
        input_dim=1280,
        hidden_size=256,
        num_layers=3,
        num_classes=41
    ).to(DEVICE)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Training on:", DEVICE)

    # ------------ Save setup --------------
    best_edit = 999
    save_path_best = "best_model.pt"
    save_path_final = "final_model.pt"

    # ------------ Train loop --------------
    from tqdm import tqdm

    # ------------ Train loop --------------
    NUM_EPOCHS = 20

    for epoch in range(NUM_EPOCHS):

        model.train()
        total_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)

        for X, y_cat, Ls, Ts in loop:
            X = X.to(DEVICE)
            Ts = Ts.to(DEVICE)

            # ---- Preprocess ----
            X_pp, Ts_new = preproc.preprocess(X, Ts)
            X_pp = X_pp.to(DEVICE)

            # ---- Forward ----
            logits = model(X_pp, Ts_new)  # (T,B,C)
            log_probs = logits.log_softmax(dim=-1)

            # ---- ctc loss ----
            optimizer.zero_grad()

            input_lengths = Ts_new.cpu()
            target_lengths = Ls.cpu()

            loss = criterion(
                log_probs,  # (T,B,C)
                y_cat,  # concatenated labels
                input_lengths,
                target_lengths
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # update tqdm bar
            loop.set_postfix(loss=loss.item())

        # ---------- Eval ----------
        train_pref, train_edit = evaluate(model, train_loader, preproc, max_batches=20)
        test_pref,  test_edit  = evaluate(model, test_loader,  preproc, max_batches=20)

        print(f"[Epoch {epoch}]")
        print(f"  Train: loss={total_loss:.3f}  pref={train_pref:.3f}  edit={train_edit:.3f}")
        print(f"  Test:              pref={test_pref:.3f}  edit={test_edit:.3f}")

        # Show one example from test set
        show_one(model, test_loader, preproc, id2phoneme, title=f"Epoch {epoch}")

        # ---------- Save best ----------
        if test_edit < best_edit:
            best_edit = test_edit
            torch.save(model.state_dict(), save_path_best)
            print(f"  >>> Saved BEST model with edit={best_edit:.3f}")

    # Save final model
    torch.save(model.state_dict(), save_path_final)
    print(f"\nTraining finished. Saved final model to {save_path_final}")

    return model


if __name__ == "__main__":
    train()
