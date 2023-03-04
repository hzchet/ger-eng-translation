from tqdm import tqdm
import wandb

from utils.mask import create_mask
from evaluating import evaluate_epoch


def train_epoch(model, optimizer, criterion, train_loader, device, tqdm_desc):
    model.train()
    losses = 0

    for src, trg in tqdm(train_loader, desc=tqdm_desc):
        src = src.to(device)
        trg = trg.to(device)

        trg_input = trg[:-1, :]

        src_mask, trg_mask, src_padding_mask, trg_padding_mask = create_mask(src, trg_input, device)

        logits = model(src, trg_input, src_mask, trg_mask, src_padding_mask, trg_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        trg_out = trg[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), trg_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item() * src.shape[0]

    return losses / len(train_loader.dataset)


def val_epoch(model, criterion, val_loader, device, tqdm_desc):
    model.val()
    losses = 0

    for src, trg in tqdm(val_loader, desc=tqdm_desc):
        src = src.to(device)
        trg = trg.to(device)

        trg_input = trg[:-1, :]

        src_mask, trg_mask, src_padding_mask, trg_padding_mask = create_mask(src, trg_input, device)

        logits = model(src, trg_input, src_mask, trg_mask, src_padding_mask, trg_padding_mask, src_padding_mask)
        
        trg_out = trg[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), trg_out.reshape(-1))
        losses += loss.item() * src.shape[0]
    
    return losses / len(val_loader.dataset)


def train(model, optimizer, criterion, train_loader, val_loader, num_epochs, device, src_transform, trg_vocab):
    for i_epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, optimizer, criterion, train_loader, device, tqdm_desc=f'Training epoch {i_epoch}/{num_epochs}')
        val_loss = val_epoch(model, criterion, val_loader, device, tqdm_desc=f'Validating epoch {i_epoch}/{num_epochs}')
        
        bleu = evaluate_epoch(model, val_loader, device, src_transform, trg_vocab, tqdm_desc=f'Translating. epoch {i_epoch}/{num_epochs}')

        wandb.log({
            'epoch': i_epoch,
            'train_loss': train_loss,
            'valid_loss': val_loss,
            'bleu': bleu
        })
