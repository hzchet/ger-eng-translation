from tqdm import tqdm
import torch
import sacrebleu

from utils.mask import generate_square_subsequent_mask
from dataloader import EOS_IDX, BOS_IDX


def greedy_decode(model, src, src_mask, max_len, start_symbol, device):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)

    for _ in range(max_len - 1):
        memory = memory.to(device)
        trg_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(device)
        out = model.decode(ys, memory, trg_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).types_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    
    return ys


def translate(model: torch.nn.Module, src_sentence: str, src_transform, trg_vocab, device):
    model.eval()
    src = src_transform(src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    trg_tokens = greedy_decode(model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX, device=device).flatten()
    return " ".join(trg_vocab.lookup_tokens(list(trg_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")


def evaluate_epoch(model, val_loader, device, src_transform, trg_vocab, tqdm_desc):
    translations = []
    trg_sentences = []
    for src, trg in tqdm(val_loader.dataset, desc=tqdm_desc):
        trg_sentences.append(trg)
        translations.append(translate(model, src, src_transform, trg_vocab))
    
    return sacrebleu.corpus_bleu(translations, trg_sentences).score
        