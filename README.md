# Automatic Speech Recognition with Conformer-CTC

<p align="center">
  <a href="#about">About</a> •
  <a href="#results">Results</a> •
  <a href="#installation">Installation</a> •
  <a href="#inference">Inference</a> •
  <a href="#training">Training</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

HSE DLA course, Homework 1 (ASR), task `hw1ASR` in [markovka17/dla](https://github.com/markovka17/dla). Conformer-CTC trained from scratch on LibriSpeech `train-clean-100`, with a hand-written CTC beam search.

Model details, training logs and experiments: [W&B report](https://forge.coreweave.com/wandb/anyapp2812/asr_hw/reports/ASR-Conformer-CTC-LibriSpeech--VmlldzoxODE0MzIwNg?accessToken=70nxuwvwxbh2s1b5rswevnw3qen1xdhf3tvl3yv0nxftnnsbat6ryvq6faczepwr)

## Results

| Dataset | Decoding | CER, % | WER, % |
|---|---|---|---|
| test-clean | argmax | 6.27 | 19.68 |
| test-clean | beam search (10) | 6.15 | 19.31 |
| test-other | argmax | 17.68 | 42.67 |
| test-other | beam search (10) | 17.48 | 42.26 |

## Installation

1. Create and activate a new environment:

   ```bash
   python3 -m venv project_env
   source project_env/bin/activate
   ```

2. Install the requirements:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the final checkpoint to `saved/conformer_ctc/model_best.pth`:

   ```bash
   python3 scripts/download_checkpoint.py
   ```

## Inference

Run inference on a directory with the following structure:

```
NameOfTheDirectoryWithUtterances
├── audio
│   ├── UtteranceID1.wav  # may be flac or mp3
│   └── ...
└── transcriptions        # ground truth, may not exist
    ├── UtteranceID1.txt
    └── ...
```

```bash
python3 inference.py datasets.test.data_dir=PATH_TO_DIR inferencer.save_path=PATH_TO_PREDICTIONS
```

Predictions are saved to `PATH_TO_PREDICTIONS/test/UtteranceID.txt`. Beam search with beam size 10 is used by default; set `inferencer.beam_size=0` for argmax decoding.

Calculate WER and CER:

```bash
python3 calc_metrics.py --gt_dir PATH_TO_DIR/transcriptions --pred_dir PATH_TO_PREDICTIONS/test
```

Evaluate on LibriSpeech test-clean and test-other (argmax and beam search metrics are printed):

```bash
python3 inference.py datasets=librispeech_test inferencer.save_path=librispeech_test
```

The demo notebook [`demo.ipynb`](demo.ipynb) runs all the steps in Google Colab for a dataset given by a Google Drive link.

## Training

The final model was trained with one command (~10 hours on a Kaggle GPU):

```bash
python3 train.py -cn=conformer_ctc
```

LibriSpeech is downloaded automatically to `data/datasets/librispeech`. To use an existing copy, put (or symlink) the parts there, e.g. `data/datasets/librispeech/train-clean-100`.

## Credits

This repository is based on the [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-green.svg)](/LICENSE)
