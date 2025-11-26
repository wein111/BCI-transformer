import torch
from torch.utils.data import DataLoader

from dataset import SpikeDataset, collate_fn
from preprocess import SpikePreprocessor
from model_GRU import LightGRUDecoder
from wfst_decoder import WFST_PyniniDecoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():

    decoder = WFST_PyniniDecoder(
        tlg_path="../languageModel/TLG.fst",
        token_path="../languageModel/tokens.txt",
        word_path="../languageModel/words.txt"
    )

    model = LightGRUDecoder(...)
    model.load_state_dict(torch.load("best_model.pt"))
    model.to(DEVICE)
    model.eval()

    pre = SpikePreprocessor(...)

    dataset = SpikeDataset(["/Users/wei/Courses/EE675/SpeechBCI-Transformer/BCI-transformer/Dataset/derived/tfRecords/t12.2022.04.28/test/chunk_0.tfrecord"])
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    X, y, Ls, Ts = next(iter(loader))
    X = X.to(DEVICE)
    Ts = Ts.to(DEVICE)

    Xp, Tp = pre.preprocess(X, Ts)
    logits = model(Xp, Tp)[0]
    log_probs = logits.log_softmax(dim=-1)

    words = decoder.decode_logits(log_probs)

    print("Decoded words:", " ".join(words))


if __name__ == "__main__":
    main()
