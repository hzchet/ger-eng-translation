import random

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


class Translations:
    def __init__(self, path_to_folder: str = None, split: str = 'train', src: str = 'de', trg: str = 'en', subset: float = 0.3):
        self.path_to_folder = path_to_folder
        
        if split not in ('train', 'val'):
            raise ValueError('"split" must be train or val')

        self.split = split
        self.src = src
        self.trg = trg
        self.subset = subset        

        self.__read_data__()

    def __read_data__(self):
        self.src_path = self.path_to_folder + '/' + self.split + f'.de-en.{self.src}'
        self.trg_path = self.path_to_folder + '/' + self.split + f'.de-en.en.{self.trg}'

        with open(self.src_path) as f:
            src_text = f.readlines()
        
        with open(self.trg_path) as f:
            trg_text = f.readlines()

        self.texts = zip(src_text, trg_text)

        if self.split == 'train':
            random.shuffle(self.texts)
            
            self.texts[0] = self.texts[0][:int(len(self.texts[0]) * self.subset)]
            self.texts[1] = self.texts[1][:int(len(self.texts[1]) * self.subset)]

    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):
        self.iter += 1
        
        if self.iter == self.__len__():
            raise StopIteration

        return self.texts[self.iter], self.texts[self.iter]
