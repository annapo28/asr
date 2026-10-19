import argparse
import urllib.request
from pathlib import Path

CHECKPOINT_URL = (
    "https://huggingface.co/anyapopova/conformer_ctc/resolve/main/model_best.pth"
)
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "saved/conformer_ctc/model_best.pth"


def main(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)
    print(f"Done: {path.stat().st_size / 2**20:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the final checkpoint")
    parser.add_argument("--url", default=CHECKPOINT_URL)
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    main(args.url, args.path)
