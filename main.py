from dataloader import get_loader
from utils import setup


def run():
    model, criterion, optimizer, train_loader, val_loader = setup()


if __name__ == '__main__':
    run()
