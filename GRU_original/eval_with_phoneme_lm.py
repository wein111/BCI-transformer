# eval_with_phoneme_lm.py
import math
import numpy as np
from collections import defaultdict, Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import tensorflow as tf
import editdistance

from preprocess_original import OriginalPreprocessor
from ID2phoneme import id2phoneme

# ===================== 设备 =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLANK_ID = 0
NUM_CLASSES = 41  # 0=blank, 1..40=phonemes


# =========================================================
# Dataset / collate：完全照 train_original.py
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

        # compute global mean/std (per dataset)
        self._compute_stats()

    def _compute_stats(self):
        allX = []
        for ex in self.samples:
            e = tf.io.parse_single_example(ex, self.feature_desc)
            allX.append(e["inputFeatures"].numpy())

        Xcat = np.concatenate(allX, axis=0)
        self.mean = Xcat.mean(axis=0)
        self.std  = Xcat.std(axis=0) + 1e-5

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        e = tf.io.parse_single_example(self.samples[idx], self.feature_desc)

        X = e["inputFeatures"].numpy()
        y = e["seqClassIDs"].numpy()
        L = int(e["nSeqElements"].numpy())
        T = int(e["nTimeSteps"].numpy())

        y = y[:L]

        # 跟 train 一样：返回 mean/std（整 dataset 的全局 mean/std）
        return X.astype(np.float32), y, T, L, self.mean, self.std


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
# Model: GRU6 — 完全照 train_original.py
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
        """
        x: (B, T, C)
        lengths: (B,) 实际长度（preprocess 后的 T_new）
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        logits = self.fc(out)        # (B, T, C)
        return logits.permute(1, 0, 2)   # (T, B, C) — 和 train 完全一致


# =========================================================
# 工具函数：切 label / 距离 / 前缀准确率
# =========================================================
def split_targets(ycat, Ls):
    """
    把 concatenated ycat 按 Ls 切回一条条序列
    ycat: (sum_L,)
    Ls  : (B,)
    """
    seqs = []
    start = 0
    for L in Ls:
        L = int(L)
        seqs.append(ycat[start:start+L])
        start += L
    return seqs


def prefix_accuracy(pred, true):
    Lp = min(len(pred), len(true))
    if Lp == 0:
        return 0.0
    return sum(pred[i] == true[i] for i in range(Lp)) / Lp


# =========================================================
# Greedy CTC decode（保持和 train 的逻辑一致）
# =========================================================
def greedy_ctc_decode(logits):
    """
    logits: (T, C) torch tensor
    返回：去重去 blank 后的 phoneme id 序列
    """
    logp = logits.log_softmax(dim=-1)
    best = logp.argmax(dim=-1).cpu().tolist()

    seq = []
    prev = -1
    for p in best:
        if p != BLANK_ID and p != prev:
            seq.append(int(p))
        prev = p
    return seq


# =========================================================
# 简单 Phoneme Bigram LM + CTC beam search
# =========================================================
class PhonemeBigramLM:
    def __init__(self, sequences, vocab_size, blank_id=0, start_token=-1, add_k=0.1):
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        self.start = start_token
        self.add_k = add_k

        self.bigram_counts = defaultdict(Counter)
        self.unigram_counts = Counter()

        for seq in sequences:
            prev = self.start
            for p in seq:
                if p == blank_id:
                    continue
                self.bigram_counts[prev][p] += 1
                self.unigram_counts[prev] += 1

        self._log_probs = {}
        for prev, counter in self.bigram_counts.items():
            total = self.unigram_counts[prev] + add_k * (vocab_size - 1)
            self._log_probs[prev] = {
                p: math.log((c + add_k) / total)
                for p, c in counter.items()
            }
            self._log_probs[prev]['<unk>'] = math.log(add_k / total)

    def log_prob(self, prev_ph, ph):
        if ph == self.blank_id:
            return 0.0
        table = self._log_probs.get(prev_ph, None)
        if table is None:
            # 完全没见过这个 prev 时，用均匀分布
            return -math.log(self.vocab_size - 1)
        return table.get(ph, table['<unk>'])


class CTCBeamSearchWithLM:
    def __init__(self, beam_size=20, lm=None, lm_weight=0.5, length_penalty=0.0):
        self.beam_size = beam_size
        self.lm = lm
        self.lm_weight = lm_weight
        self.length_penalty = length_penalty

    def decode(self, log_probs_np):
        """
        log_probs_np: (T, C) numpy array (log-softmax 后的对数概率)
        """
        T, C = log_probs_np.shape

        # prefix -> (pb, pnb, last_nonblank)
        beam = {(): (0.0, -np.inf, None)}

        for t in range(T):
            lp = log_probs_np[t]  # (C,)
            new_beam = {}

            for prefix, (pb, pnb, last_nb) in beam.items():
                # 1) blank
                lp_blank = lp[BLANK_ID]
                if prefix not in new_beam:
                    new_beam[prefix] = (-np.inf, -np.inf, last_nb)
                pb_new, pnb_new, last_nb_new = new_beam[prefix]
                pb_new = np.logaddexp(pb_new, pb + lp_blank)
                pb_new = np.logaddexp(pb_new, pnb + lp_blank)
                new_beam[prefix] = (pb_new, pnb_new, last_nb_new)

                # 2) non-blank
                for c in range(1, C):
                    lp_c = lp[c]
                    lm_bonus = 0.0
                    if self.lm is not None:
                        need_lm = (len(prefix) == 0) or (c != prefix[-1])
                        if need_lm:
                            cond_prev = last_nb if last_nb is not None else self.lm.start
                            lm_bonus = self.lm_weight * self.lm.log_prob(cond_prev, c)

                    if len(prefix) > 0 and c == prefix[-1]:
                        # 重复，只能从 blank
                        new_prefix = prefix
                        if new_prefix not in new_beam:
                            new_beam[new_prefix] = (-np.inf, -np.inf, last_nb)
                        pb_old, pnb_old, last_nb_old = new_beam[new_prefix]
                        pnb_old = np.logaddexp(pnb_old, pb + lp_c + lm_bonus)
                        new_beam[new_prefix] = (pb_old, pnb_old, c)
                    else:
                        # 扩展新 label
                        new_prefix = prefix + (c,)
                        if new_prefix not in new_beam:
                            new_beam[new_prefix] = (-np.inf, -np.inf, None)
                        pb_old, pnb_old, last_nb_old = new_beam[new_prefix]
                        pnb_old = np.logaddexp(pnb_old, pb + lp_c + lm_bonus)
                        pnb_old = np.logaddexp(pnb_old, pnb + lp_c + lm_bonus)
                        new_beam[new_prefix] = (pb_old, pnb_old, c)

            # beam 剪枝
            items = []
            for prefix, (pb, pnb, last_nb) in new_beam.items():
                score = max(pb, pnb)
                items.append((score, prefix, pb, pnb, last_nb))
            items.sort(key=lambda x: x[0], reverse=True)
            beam = {}
            for score, prefix, pb, pnb, last_nb in items[: self.beam_size]:
                beam[prefix] = (pb, pnb, last_nb)

        # 最后选 score 最大的 prefix
        best_prefix = None
        best_score = -np.inf
        for prefix, (pb, pnb, last_nb) in beam.items():
            score = np.logaddexp(pb, pnb) + self.length_penalty * len(prefix)
            if score > best_score:
                best_score = score
                best_prefix = prefix

        return list(best_prefix) if best_prefix is not None else []


# =========================================================
# 构建 LM（从训练 label）
# =========================================================
def build_phoneme_lm(train_loader, preproc, max_batches=200):
    print("Building phoneme bigram LM from training labels ...")
    all_seqs = []

    for bi, (X, ycat, Ls, Ts, means, stds) in enumerate(
        tqdm(train_loader, desc="Collect LM data")
    ):
        if bi >= max_batches:
            break

        # 把 ycat/Ls 切成每条样本的 true phoneme 序列
        ycat_np = ycat.numpy()
        Ls_list = Ls.numpy()
        true_seqs = split_targets(ycat_np, Ls_list)

        # 去掉非法 id / blank
        for s in true_seqs:
            clean = [int(x) for x in s if 1 <= int(x) < NUM_CLASSES]
            if len(clean) > 0:
                all_seqs.append(clean)

    lm = PhonemeBigramLM(
        sequences=all_seqs,
        vocab_size=NUM_CLASSES,
        blank_id=BLANK_ID,
        start_token=-1,
        add_k=0.1,
    )
    print(f"  Collected {len(all_seqs)} sequences for LM.")
    return lm


# =========================================================
# 评估（greedy 和 beam+LM）
# =========================================================
def evaluate(model, loader, preproc, decoder_lm=None, max_batches=100):
    model.eval()

    total_pref_g = 0.0
    total_edit_g = 0.0

    total_pref_lm = 0.0
    total_edit_lm = 0.0

    total_cnt = 0

    beam_with_lm = None
    if decoder_lm is not None:
        beam_with_lm = CTCBeamSearchWithLM(
            beam_size=20, lm=decoder_lm, lm_weight=0.7
        )

    with torch.no_grad():
        for bi, (X, ycat, Ls, Ts, means, stds) in enumerate(
            tqdm(loader, desc="Eval")
        ):
            if bi >= max_batches:
                break

            X_np = X.numpy()
            B = X_np.shape[0]

            # ---------- preprocess：完全照 train ----------
            X_new_list = []
            T_new_list = []
            for i in range(B):
                Xi = X_np[i, : Ts[i]]
                Xi, T_new = preproc(Xi, means[i], stds[i])
                X_new_list.append(Xi)
                T_new_list.append(T_new)

            Tmax = max(T_new_list)
            feat_dim = X_new_list[0].shape[1]

            X_pad = np.zeros((B, Tmax, feat_dim), dtype=np.float32)
            for i in range(B):
                X_pad[i, : T_new_list[i]] = X_new_list[i]

            X_pad = torch.tensor(X_pad, dtype=torch.float32).to(DEVICE)
            T_new = torch.tensor(T_new_list, dtype=torch.long).to(DEVICE)

            logits = model(X_pad, T_new)  # (T_max', B, C)

            # 切 label 序列
            ycat_list = ycat.numpy().tolist()
            Ls_list = Ls.numpy().tolist()
            true_seqs = []
            offset = 0
            for L in Ls_list:
                true_seqs.append(ycat_list[offset : offset + L])
                offset += L

            # 每条样本独立评估
            for b in range(B):
                Tlen = T_new_list[b]
                logit_b = logits[:Tlen, b, :]  # (Tlen, C)
                true = true_seqs[b]

                # greedy
                pred_g = greedy_ctc_decode(logit_b)
                pref_g = prefix_accuracy(pred_g, true)
                edit_g = editdistance.eval(pred_g, true) / max(len(true), 1)

                total_pref_g += pref_g
                total_edit_g += edit_g

                # beam + LM
                if beam_with_lm is not None:
                    logp_np = logit_b.log_softmax(dim=-1).detach().cpu().numpy()
                    pred_lm = beam_with_lm.decode(logp_np)
                    pref_lm = prefix_accuracy(pred_lm, true)
                    edit_lm = editdistance.eval(pred_lm, true) / max(len(true), 1)

                    total_pref_lm += pref_lm
                    total_edit_lm += edit_lm

                total_cnt += 1

    if total_cnt == 0:
        return (0, 0), (0, 0)

    avg_g = (total_pref_g / total_cnt, total_edit_g / total_cnt)
    avg_lm = (total_pref_lm / total_cnt, total_edit_lm / total_cnt) if decoder_lm else (0, 0)
    return avg_g, avg_lm


# =========================================================
# 打印一些样本对比
# =========================================================
def show_samples(model, loader, preproc, decoder_lm=None, num_batches=2):
    model.eval()
    beam_with_lm = None
    if decoder_lm is not None:
        beam_with_lm = CTCBeamSearchWithLM(
            beam_size=20, lm=decoder_lm, lm_weight=0.7
        )

    with torch.no_grad():
        for bi, (X, ycat, Ls, Ts, means, stds) in enumerate(loader):
            if bi >= num_batches:
                break

            X_np = X.numpy()
            B = X_np.shape[0]

            # preprocess
            X_new_list = []
            T_new_list = []
            for i in range(B):
                Xi = X_np[i, : Ts[i]]
                Xi, T_new = preproc(Xi, means[i], stds[i])
                X_new_list.append(Xi)
                T_new_list.append(T_new)

            Tmax = max(T_new_list)
            feat_dim = X_new_list[0].shape[1]
            X_pad = np.zeros((B, Tmax, feat_dim), dtype=np.float32)
            for i in range(B):
                X_pad[i, : T_new_list[i]] = X_new_list[i]

            X_pad = torch.tensor(X_pad, dtype=torch.float32).to(DEVICE)
            T_new = torch.tensor(T_new_list, dtype=torch.long).to(DEVICE)
            logits = model(X_pad, T_new)  # (T',B,C)

            # 切 label
            ycat_list = ycat.numpy().tolist()
            Ls_list = Ls.numpy().tolist()
            true_seqs = []
            offset = 0
            for L in Ls_list:
                true_seqs.append(ycat_list[offset : offset + L])
                offset += L

            for b in range(B):
                Tlen = T_new_list[b]
                logit_b = logits[:Tlen, b, :]
                true = true_seqs[b]

                greedy_ids = greedy_ctc_decode(logit_b)
                logp_np = logit_b.log_softmax(dim=-1).detach().cpu().numpy()
                lm_ids = beam_with_lm.decode(logp_np) if beam_with_lm else []

                print(f"\n===== Batch {bi} Sample {b} =====")
                print("True IDs :", true)
                print("True PH  :", [id2phoneme.get(i, '?') for i in true])

                print("Greedy IDs:", greedy_ids)
                print("Greedy PH :", [id2phoneme.get(i, '?') for i in greedy_ids])

                if beam_with_lm is not None:
                    print("Beam+LM IDs:", lm_ids)
                    print("Beam+LM PH :", [id2phoneme.get(i, '?') for i in lm_ids])
                print("========================================")


# =========================================================
# main
# =========================================================
def main():
    print("Device:", DEVICE)

    # ===== 路径（按你训练时的来，自己切换本地 / Colab） =====
    # 本地 Mac 版本（如果你有本地数据）:
    # BASE = "/Users/wei/Courses/EE675/SpeechBCI-Transformer/BCI-transformer/Dataset/derived/tfRecords"
    # MODEL_PATH = "/Users/wei/Courses/EE675/SpeechBCI-Transformer/BCI-transformer/GRU_original/best_model.pt"

    # Colab 版本：
    BASE = "/Users/wei/Courses/EE675/SpeechBCI-Transformer/BCI-transformer/Dataset/derived/tfRecords"
    MODEL_PATH = "/Users/wei/Courses/EE675/SpeechBCI-Transformer/BCI-transformer/GRU_original/best_model.pt"

    DATES = [
        "t12.2022.04.28", "t12.2022.05.05", "t12.2022.05.17",
        "t12.2022.05.19", "t12.2022.05.24", "t12.2022.05.26",
        "t12.2022.06.02", "t12.2022.06.07", "t12.2022.06.14",
        "t12.2022.06.16", "t12.2022.06.21", "t12.2022.06.23",
        "t12.2022.06.28", "t12.2022.07.05", "t12.2022.07.14",
        "t12.2022.07.21", "t12.2022.07.27", "t12.2022.07.29",
    ]

    train_paths = [f"{BASE}/{d}/train/chunk_0.tfrecord" for d in DATES]
    test_paths  = [f"{BASE}/{d}/test/chunk_0.tfrecord"  for d in DATES]

    train_ds = SpikeDataset(train_paths)
    test_ds  = SpikeDataset(test_paths)

    train_loader = DataLoader(train_ds, batch_size=6, shuffle=True,  collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=6, shuffle=False, collate_fn=collate_fn)

    # preprocess — 必须和 train 一样
    preproc = OriginalPreprocessor(
        smooth_sigma=1.5,
        stack_k=5,
        stack_stride=2,
        subsample_factor=3,
    )

    # model — 跟 train 完全一致
    model = GRU6(
        input_dim=256 * 5,
        hidden=256,
        num_layers=6,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    print(f"Loading model from: {MODEL_PATH}")
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    # ====== 构建 phoneme bigram LM ======
    lm = build_phoneme_lm(train_loader, preproc, max_batches=1000)

    # ====== Evaluate ======
    print("\n==== TRAIN SET ====")
    (gp_tr, ge_tr), (lp_tr, le_tr) = evaluate(
        model, train_loader, preproc, decoder_lm=lm, max_batches=300
    )
    print(f"Greedy pref={gp_tr:.3f}, PER={ge_tr:.3f}")
    print(f"Beam+LM pref={lp_tr:.3f}, PER={le_tr:.3f}")

    print("\n==== TEST SET ====")
    (gp_te, ge_te), (lp_te, le_te) = evaluate(
        model, test_loader, preproc, decoder_lm=lm, max_batches=300
    )
    print(f"Greedy pref={gp_te:.3f}, PER={ge_te:.3f}")
    print(f"Beam+LM pref={lp_te:.3f}, PER={le_te:.3f}")

    # ====== 一些样本 ======
    print("\n==== Some sample decodes on TEST ====")
    show_samples(model, test_loader, preproc, decoder_lm=lm, num_batches=1)


if __name__ == "__main__":
    main()

#111111