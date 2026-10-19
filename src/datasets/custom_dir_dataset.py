from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset


class CustomDirDataset(BaseDataset):
    """
    Dataset for a directory with `audio/` and optional `transcriptions/`
    subdirectories. Utterance ID is the file name without extension.
    """

    def __init__(self, data_dir, *args, **kwargs):
        data_dir = Path(data_dir)
        index = []
        for path in sorted((data_dir / "audio").iterdir()):
            if path.suffix not in (".wav", ".flac", ".mp3"):
                continue
            transcription = data_dir / "transcriptions" / f"{path.stem}.txt"
            text = transcription.read_text().strip() if transcription.exists() else ""
            info = torchaudio.info(str(path))
            index.append(
                {
                    "path": str(path),
                    "text": text,
                    "audio_len": info.num_frames / info.sample_rate,
                }
            )
        super().__init__(index, *args, **kwargs)
