import torch

from dataloader import get_loader, PAD_IDX
from models.architectures import Seq2SeqTransformer


def parse(config):
    if 'num_epochs' not in config:
        raise ValueError('"num_epochs" parameter is not specified')

    device = torch.device(config['device'])

    train_loader, src_vocab_size, trg_vocab_size, src_transform, trg_vocab = get_loader(
        split='train',
        batch_size=config['batch_size'],
        path_to_folder=config['path_to_data'],
        src=config['src_ln'],
        trg=config['trg_ln'],
        subset=config['train_subset'],
        min_freq=config['min_freq']
    )
    val_loader, _, _, _, _ = get_loader(
        split='val',
        batch_size=config['batch_size'],
        path_to_folder=config['path_to_data'],
        src=config['src_ln'],
        trg=config['trg_ln'],
        min_freq=config['min_freq']
    )

    architecture = config['architecture']
    if architecture['name'] == 'transformer':
        params = architecture['params']
        params['src_vocab_size'] = src_vocab_size
        params['trg_vocab_size'] = trg_vocab_size
        
        model = Seq2SeqTransformer(**params)
        
        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        
        model = model.to(device)
    else:
        raise NotImplementedError("Currently supports only transformer architecture")

    if config['criterion'] == 'CrossEntropyLoss':
        criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    else:
        raise NotImplementedError("Currently supports only CrossEntropyLoss as a critetion")
    
    if config['optimizer']['name'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), **config['optimizer']['params'])
    else:
        raise NotImplementedError("Currently supports only Adam as an optimizer")

    return model, criterion, optimizer, train_loader, val_loader, config['num_epochs'], device, src_transform, trg_vocab
