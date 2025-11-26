# eval_beam_ctc.py

import torch
from torch.utils.data import DataLoader
import numpy as np

from dataset import SpikeDataset, collate_fn
from preprocessor import SpikePreprocessor
from model_gru_light import LightGRUDecoder
from beam_search_decoder import BeamSearchCTCDecoder
from ID2phoneme import id2phoneme   # 确保路径正确

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------
# Edit distance (Levenshtein)
# ---------------------------
def edit_distance(a, b):
    """
    a, b: list of int
    return: integer edit distance
    """
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
# Evaluate on full dataloader
# ---------------------------
@torch.no_grad()
def evaluate_beam(model, loader, preproc, max_batches=None, print_first_n=5):
    model.eval()
    decoder = BeamSearchCTCDecoder(beam_width=10, blank=0)

    total_pref = 0.0
    total_edit = 0.0
    total_cnt = 0

    printed = 0

    for bi, (X, y_cat, Ls, Ts) in enumerate(loader):
        if (max_batches is not None) and (bi >= max_batches):
            break

        X = X.to(DEVICE)
        Ts = Ts.to(DEVICE)

        # 预处理
        Xp, Tp = preproc.preprocess(X, Ts)
        Xp = Xp.to(DEVICE)

        # 模型前向：logits (T,B,C)
        logits = model(Xp, Tp)
        # 这里 batch_size=1 时，取[:,0,:]
        logit = logits[:, 0, :]  # (T, C)

        # Beam search CTC
        pred_ids = decoder.decode(logit)
        L_true = Ls[0].item()
        true_ids = y_cat[:L_true].tolist()

        # prefix accuracy
        Lp = min(len(pred_ids), len(true_ids))
        if Lp > 0:
            pref = sum(pred_ids[i] == true_ids[i] for i in range(Lp)) / Lp
        else:
            pref = 0.0

        # edit rate (normalized by true length)
        ed = edit_distance(pred_ids, true_ids)
        edit_rate = ed / max(1, len(true_ids))

        total_pref += pref
        total_edit += edit_rate
        total_cnt += 1

        # 打印前几条看看
        if printed < print_first_n:
            printed += 1
            print(f"\n===== Sample {printed} (batch {bi}) =====")
            print("Pred IDs:", pred_ids)
            print("True IDs:", true_ids)
            print("Pred PH :", [id2phoneme.get(i, "?") for i in pred_ids])
            print("True PH :", [id2phoneme.get(i, "?") for i in true_ids])
            print(f"Prefix acc: {pref:.3f}, Edit rate: {edit_rate:.3f}")
            print("=======================================")

    if total_cnt == 0:
        return 0.0, 0.0

    return total_pref / total_cnt, total_edit / total_cnt


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

    # Dataset + Dataloader
    test_dataset = SpikeDataset(test_paths)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,            # 方便打印和对齐
        shuffle=False,
        collate_fn=collate_fn
    )

    # Preprocessor (保持和训练时一致)
    preproc = SpikePreprocessor(stack_k=5, stack_stride=1, subsample_factor=2)

    # Model
    model = LightGRUDecoder(
        input_dim=1280,
        hidden_size=256,
        num_layers=3,
        num_classes=41
    ).to(DEVICE)

    # 加载你训练好的权重
    state = torch.load("best_model.pt", map_location=DEVICE)
    model.load_state_dict(state)
    print("Loaded model from best_model.pt")
    print("Evaluating on test set ...")

    # Evaluate
    avg_pref, avg_edit = evaluate_beam(
        model, test_loader, preproc,
        max_batches=None,      # None → 全部 test
        print_first_n=5        # 多看几条
    )

    print("\n========== Test Set Summary ==========")
    print(f"Average prefix accuracy: {avg_pref:.3f}")
    print(f"Average edit rate      : {avg_edit:.3f}")
    print("=======================================")


if __name__ == "__main__":
    main()
