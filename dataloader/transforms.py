from typing import List

import torch
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from translations import Translations


UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


def yield_tokens(tokenizer, data_iter: Translations, language: str) -> List[str]:
    language_index = {data_iter.src: 0, data_iter.trg: 1}

    for data_sample in data_iter:
        yield tokenizer(data_sample[language_index[language]])


def get_vocab_transform(path_to_folder, tokenizer, ln, min_freq=1):
    train_iter = Translations(path_to_folder, split='train')
    vocab = build_vocab_from_iterator(
        yield_tokens(tokenizer, train_iter, ln),
        min_freq=min_freq,
        specials=special_symbols,
        special_first=True
    )
    vocab.set_default_index(UNK_IDX)
    return vocab


def sequential_transforms(*transforms):
    """
    helper function to club together sequential operations
    """
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    
    return func


def tensor_transform(token_ids: List[int]):
    """
    function to add BOS/EOS and create tensor for input sequence indices
    """
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))


def collate_fn(text_transform):
    def inner(batch):
        """
        function to collate data samples into batch tesors
        """
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(text_transform[0](src_sample.rstrip("\n")))
            tgt_batch.append(text_transform[1](tgt_sample.rstrip("\n")))

        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
        return src_batch, tgt_batch

    return inner

