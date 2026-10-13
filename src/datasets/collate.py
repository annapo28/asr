import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    specs = [item["spectrogram"][0].T for item in dataset_items]
    texts = [item["text_encoded"][0] for item in dataset_items]
    audios = [item["audio"][0] for item in dataset_items]

    return {
        "spectrogram": pad_sequence(specs).permute(1, 2, 0),
        "spectrogram_length": torch.tensor([len(s) for s in specs]),
        "text_encoded": pad_sequence(texts).T,
        "text_encoded_length": torch.tensor([len(t) for t in texts]),
        "audio": pad_sequence(audios).T,
        "text": [item["text"] for item in dataset_items],
        "audio_path": [item["audio_path"] for item in dataset_items],
    }
