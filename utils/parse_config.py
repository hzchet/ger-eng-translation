from dataloader import get_loader


def parse(config):
    train_loader, src_vocab_size, trg_vocab_size = get_loader(
        split='train',
        batch_size=config['batch_size'],
        path_to_folder=config['path_to_data'],
        src=config['src_ln'],
        trg=config['trg_ln'],
        subset=config['train_subset']
    )
    val_loader, _, _ = get_loader(
        split='val',
        batch_size=config['batch_size'],
        path_to_folder=config['path_to_data'],
        src=config['src_ln'],
        trg=config['trg_ln']
    )

    