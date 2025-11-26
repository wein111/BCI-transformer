# eval_greedy_ctc.py

import torch
from torch.utils.data import DataLoader
import numpy as np

from dataset import SpikeDataset, collate_fn
from preprocess import SpikePreprocessor
from model_GRU import LightGRUDecoder
from ID2phoneme import id2phoneme  # 确保路径正确

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------
# Edit distance (Levenshtein)
# ---------------------------
def edit_distance(a, b):
    la, lb = len(a), len(b)
    dp = np.zeros((la + 1, lb + 1), dtype=int)
    dp[:, 0] = np.arange(la + 1)
    dp[0, :] = np.arange(lb + 1)
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,      # delete
                dp[i, j - 1] + 1,      # insert
                dp[i - 1, j - 1] + cost  # substitute
            )
    return dp[la, lb]


# ---------------------------
# 最简单靠谱的 greedy CTC 解码
# ---------------------------
def greedy_ctc_decode(logit, blank=0):
    """
    logit: (T, C) torch tensor (未必 log_softmax，没关系)
    返回：去掉 blank + 合并重复后的 label 序列
    """
    # 先取 argmax per frame
    pred = logit.argmax(dim=-1).tolist()  # [T]
    out = []
    prev = blank
    for p in pred:
        if p != blank and p != prev:
            out.append(p)
        prev = p
    return out


# ---------------------------
# Evaluate on full dataloader
# ---------------------------
@torch.no_grad()
def evaluate_greedy(model, loader, preproc, max_batches=None, print_first_n=5):
    model.eval()

    total_pref = 0.0
    total_edit = 0.0
    total_cnt = 0

    total_pred_len = 0
    total_true_len = 0

    printed = 0

    for bi, (X, y_cat, Ls, Ts) in enumerate(loader):
        if (max_batches is not None) and (bi >= max_batches):
            break

        X = X.to(DEVICE)
        Ts = Ts.to(DEVICE)

        # 预处理（保持和训练时一致）
        Xp, Tp = preproc.preprocess(X, Ts)
        Xp = Xp.to(DEVICE)

        # 模型输出 logits: (T, B, C)
        logits = model(Xp, Tp)
        logit = logits[:, 0, :]  # (T, C), batch=1

        # ----- Greedy CTC decode -----
        pred_ids = greedy_ctc_decode(logit, blank=0)

        L_true = Ls[0].item()
        true_ids = y_cat[:L_true].tolist()

        # prefix accuracy
        Lp = min(len(pred_ids), len(true_ids))
        if Lp > 0:
            pref = sum(pred_ids[i] == true_ids[i] for i in range(Lp)) / Lp
        else:
            pref = 0.0

        # edit rate
        ed = edit_distance(pred_ids, true_ids)
        edit_rate = ed / max(1, len(true_ids))

        total_pref += pref
        total_edit += edit_rate
        total_cnt += 1

        total_pred_len += len(pred_ids)
        total_true_len += len(true_ids)

        # 打印前几条看看
        if printed < print_first_n:
            printed += 1
            print(f"\n===== Sample {printed} (batch {bi}) =====")
            print("Pred IDs:", pred_ids)
            print("True IDs:", true_ids)
            print("Pred PH :", [id2phoneme.get(i, "?") for i in pred_ids])
            print("True PH :", [id2phoneme.get(i, "?") for i in true_ids])
            print(f"Prefix acc: {pref:.3f}, Edit rate: {edit_rate:.3f}")
            print("len(pred)={len_p}, len(true)={len_t}".format(
                len_p=len(pred_ids), len_t=len(true_ids)
            ))
            print("=======================================")

    if total_cnt == 0:
        return 0.0, 0.0, 0.0, 0.0

    avg_pref = total_pref / total_cnt
    avg_edit = total_edit / total_cnt
    avg_pred_len = total_pred_len / total_cnt
    avg_true_len = total_true_len / total_cnt

    return avg_pref, avg_edit, avg_pred_len, avg_true_len


# ---------------------------
# main
# ---------------------------
def main():
    BASE = "/Users/wei/Courses/EE675/SpeechBCI-Transformer/BCI-transformer/Dataset/derived/tfRecords"
    TEST_DATES = [
        "t12.2022.04.28", "t12.2022.05.05", "t12.2022.05.17",
        "t12.2022.05.19", "t12.2022.05.24", "t12.2022.05.26",
        "t12.2022.06.02", "t12.2022.06.07", "t12.2022.06.14",
        "t12.2022.06.16", "t12.2022.06.21", "t12.2022.06.23"
    ]

    test_paths = [f"{BASE}/{d}/test/chunk_0.tfrecord" for d in TEST_DATES]

    # Dataset + Loader
    test_dataset = SpikeDataset(test_paths)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Preprocessor（和训练时保持一致）
    preproc = SpikePreprocessor(stack_k=5, stack_stride=1, subsample_factor=2)

    # Model（结构必须和训练时完全一致）
    model = LightGRUDecoder(
        input_dim=1280,
        hidden_size=256,
        num_layers=3,
        num_classes=41
    ).to(DEVICE)

    state = torch.load("best_model.pt", map_location=DEVICE)
    model.load_state_dict(state)
    print("Loaded model from best_model.pt")
    print("Evaluating on test set with GREEDY CTC ...")

    avg_pref, avg_edit, avg_pred_len, avg_true_len = evaluate_greedy(
        model, test_loader, preproc,
        max_batches=None,
        print_first_n=5
    )

    print("\n========== Test Set Summary (Greedy CTC) ==========")
    print(f"Average prefix accuracy : {avg_pref:.3f}")
    print(f"Average edit rate (PER) : {avg_edit:.3f}")
    print(f"Avg pred length         : {avg_pred_len:.1f}")
    print(f"Avg true length         : {avg_true_len:.1f}")
    print("===================================================")


if __name__ == "__main__":
    main()
