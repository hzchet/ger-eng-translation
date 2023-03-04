from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer

from .transforms import *
from .translations import Translations


def get_loader(split: str = 'train', batch_size: int = 128, path_to_folder: str = 'data', src='de', trg='en', subset=0.3):
    data = Translations(path_to_folder, split=split, subset=subset)
    space_tokenizer = get_tokenizer()

    transform = []
    vocab_sizes = []
    for i, ln in enumerate((src, trg)):
        vocab_transform = get_vocab_transform(path_to_folder, space_tokenizer, ln)
        transform[i] = sequential_transforms(
            space_tokenizer,
            vocab_transform,
            tensor_transform
        )
        vocab_sizes[i] = len(vocab_transform)

    return DataLoader(
        data, 
        batch_size=batch_size,
        collate_fn=collate_fn(transform)
    ), *vocab_sizes
