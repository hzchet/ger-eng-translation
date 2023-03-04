from utils import setup
from models.training import train
from models.evaluating import translate


def make_submission(model, src_transform, trg_vocab, device, submission_name: str = 'baseline', path_to_test_data: str = 'data/test1.de-en.de'):
    submission_path = f'submissions/{submission_name}.de-en.en'    
    
    translated = []

    with open(path_to_test_data, 'r') as f:
        for sentence in f.readline():
            translated.append(translate(model, sentence, src_transform, trg_vocab, device))
    
    with open(submission_path, 'w') as f:
        for sentence in translated:
            f.write(sentence + '\n')


def run():
    model, criterion, optimizer, train_loader, val_loader, num_epochs, device, src_transform, trg_vocab = setup()
    
    train(
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        num_epochs,
        device,
        src_transform,
        trg_vocab
    )

    make_submission(model, src_transform, trg_vocab, device)


if __name__ == '__main__':
    run()
