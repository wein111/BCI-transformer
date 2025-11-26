# beam_search_decoder.py
import numpy as np

class BeamSearchCTCDecoder:
    """
    Beam Search Decoder for CTC.
    Adapted to use numpy.logaddexp instead of math.logaddexp (for Python <3.11)
    """

    def __init__(self, beam_width=10, blank=0):
        self.beam_width = beam_width
        self.blank = blank

    # ----------------------------------------------------
    # Main entry
    # ----------------------------------------------------
    def decode(self, log_probs):
        """
        log_probs: (T, C) numpy array or torch.tensor
                   T = time steps
                   C = number of classes
        """
        if "torch" in str(type(log_probs)):
            log_probs = log_probs.detach().cpu().numpy()

        T, C = log_probs.shape

        # Each beam entry = (prefix_tuple, (p_blank, p_nonblank))
        beams = {(): (0.0, -np.inf)}   # blank prob=0, nonblank=-inf

        for t in range(T):
            next_beams = {}

            for prefix, (pb, pnb) in beams.items():
                for c in range(C):
                    p = log_probs[t, c]

                    if c == self.blank:
                        # Extend with blank
                        nb_pb = np.logaddexp(pb + p, pnb + p)
                        self._add_beam(next_beams, prefix, (nb_pb, -np.inf))
                    else:
                        end = prefix[-1] if len(prefix) > 0 else None

                        if c == end:
                            # repeated last label
                            nb_pnb = np.logaddexp(pnb + p, -np.inf)
                            new_prefix = prefix
                            self._add_beam(next_beams, new_prefix, ( -np.inf, nb_pnb ))
                        else:
                            # extend prefix
                            new_prefix = prefix + (c,)
                            nb_pnb = np.logaddexp(pb + p, pnb + p)
                            self._add_beam(next_beams, new_prefix, ( -np.inf, nb_pnb ))

            # prune beams
            beams = self._prune(next_beams)

        # Select best beam by total probability
        best_prefix = max(beams.items(), key=lambda x: np.logaddexp(x[1][0], x[1][1]))[0]
        return list(best_prefix)

    # ----------------------------------------------------
    def _add_beam(self, beams, prefix, new_probs):
        """
        beams[prefix] = (p_blank, p_nonblank)
        """
        if prefix not in beams:
            beams[prefix] = new_probs
        else:
            old_pb, old_pnb = beams[prefix]
            new_pb, new_pnb = new_probs

            beams[prefix] = (
                np.logaddexp(old_pb, new_pb),
                np.logaddexp(old_pnb, new_pnb)
            )

    # ----------------------------------------------------
    def _prune(self, beams):
        """
        Keep top k beams
        """
        # score = logaddexp(pb, pnb)
        scored = sorted(
            beams.items(),
            key=lambda x: np.logaddexp(x[1][0], x[1][1]),
            reverse=True
        )
        return dict(scored[:self.beam_width])
