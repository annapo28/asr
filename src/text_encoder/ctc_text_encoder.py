import re
from collections import defaultdict
from string import ascii_lowercase

import numpy as np
import torch


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.blank_ind = 0

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.LongTensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        chars = []
        prev = self.blank_ind
        for ind in inds:
            ind = int(ind)
            if ind != prev and ind != self.blank_ind:
                chars.append(self.ind2char[ind])
            prev = ind
        return "".join(chars).strip()

    def ctc_beam_search(self, probs, beam_size=10, prune=1e-3) -> str:
        """
        CTC prefix beam search.

        Args:
            probs (np.ndarray): (T, V) per-frame token probabilities.
            beam_size (int): number of hypotheses kept after each frame.
            prune (float): tokens with lower probability on a frame are skipped.
        Returns:
            text (str): the most probable decoded text.
        """
        beams = {("", self.EMPTY_TOK): 1.0}
        for frame in probs:
            candidates = np.nonzero(frame >= min(prune, frame.max()))[0]
            new_beams = defaultdict(float)
            for (prefix, last_char), prob in beams.items():
                for ind in candidates:
                    char = self.ind2char[int(ind)]
                    if char == self.EMPTY_TOK:
                        key = (prefix, self.EMPTY_TOK)
                    elif char == last_char:
                        key = (prefix, last_char)
                    else:
                        key = (prefix + char, char)
                    new_beams[key] += prob * frame[ind]
            beams = dict(sorted(new_beams.items(), key=lambda x: -x[1])[:beam_size])
            total = sum(beams.values())
            beams = {key: prob / total for key, prob in beams.items()}

        scores = defaultdict(float)
        for (prefix, _), prob in beams.items():
            scores[prefix.strip()] += prob
        return max(scores, key=scores.get)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
