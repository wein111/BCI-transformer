import pynini
from pynini.lib import rewrite
import numpy as np
import torch

class WFST_PyniniDecoder:
    def __init__(self, tlg_path, token_path, word_path):
        print("Loading WFST with Pynini...")
        self.TLG = pynini.Fst.read(tlg_path)

        # token (CTC label) mapping
        self.id2token = {}
        with open(token_path, "r") as f:
            for line in f:
                sym, idx = line.strip().split()
                self.id2token[int(idx)] = sym

        # word mapping
        self.id2word = {}
        with open(word_path, "r") as f:
            for line in f:
                sym, idx = line.strip().split()
                self.id2word[int(idx)] = sym

    def logits_to_fst(self, logit: np.ndarray):
        """
        Convert T×C logits (probabilities or log probs) into a linear-chain FST.
        """

        T, C = logit.shape
        g = pynini.Fst()
        start = g.add_state()
        g.set_start(start)
        prev = start

        for t in range(T):
            topk = np.argsort(-logit[t])[:6]  # pick top-6 tokens

            cur = g.add_state()
            for tok in topk:
                weight = -float(logit[t][tok])
                g.add_arc(prev, pynini.Arc(tok, tok, weight, cur))

            prev = cur

        g.set_final(prev)
        return g

    def decode_logits(self, log_probs: torch.Tensor):
        """
        log_probs: (T, C) torch tensor (log-softmax output)
        """
        log_np = log_probs.cpu().numpy()
        emission_fst = self.logits_to_fst(log_np)

        # Compose emission with TLG graph
        composed = emission_fst @ self.TLG

        # Viterbi (shortest path)
        shortest = pynini.shortestpath(composed).optimize()

        # Extract words (output labels)
        words = []
        for state in shortest.states():
            for arc in shortest.arcs(state):
                olab = arc.olabel
                if olab in self.id2word:
                    words.append(self.id2word[olab])

        return words
