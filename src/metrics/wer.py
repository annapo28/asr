from typing import List

from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_wer


class ArgmaxWERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def decode(self, log_probs: Tensor) -> str:
        return self.text_encoder.ctc_decode(log_probs.argmax(-1))

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        wers = []
        log_probs = log_probs.detach().cpu()
        for log_prob, length, target_text in zip(log_probs, log_probs_length, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.decode(log_prob[: int(length)])
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)


class BeamSearchWERMetric(ArgmaxWERMetric):
    def __init__(self, text_encoder, beam_size=10, *args, **kwargs):
        super().__init__(text_encoder, *args, **kwargs)
        self.beam_size = beam_size

    def decode(self, log_probs: Tensor) -> str:
        return self.text_encoder.ctc_beam_search(
            log_probs.exp().numpy(), self.beam_size
        )
