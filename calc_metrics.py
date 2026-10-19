import argparse
from pathlib import Path

from src.metrics.utils import calc_cer, calc_wer
from src.text_encoder import CTCTextEncoder


def main(gt_dir, pred_dir):
    wers, cers = [], []
    for gt_path in sorted(Path(gt_dir).glob("*.txt")):
        pred_path = Path(pred_dir) / gt_path.name
        target = CTCTextEncoder.normalize_text(gt_path.read_text().strip())
        pred = pred_path.read_text().strip() if pred_path.exists() else ""
        pred = CTCTextEncoder.normalize_text(pred)
        wers.append(calc_wer(target, pred))
        cers.append(calc_cer(target, pred))

    print(f"Utterances: {len(wers)}")
    print(f"WER: {100 * sum(wers) / len(wers):.2f}")
    print(f"CER: {100 * sum(cers) / len(cers):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate WER and CER")
    parser.add_argument(
        "--gt_dir", required=True, help="directory with ground truth .txt"
    )
    parser.add_argument(
        "--pred_dir", required=True, help="directory with predicted .txt"
    )
    args = parser.parse_args()
    main(args.gt_dir, args.pred_dir)