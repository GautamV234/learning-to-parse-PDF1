from os.path import join
import pickle as pkl
from collections import Counter
import argparse
import torch

START_TOKEN = 0
PAD_TOKEN = 1
END_TOKEN = 2
UNK_TOKEN = 3


class Vocab(object):
    def __init__(self):
        self.sign2id = {"<s>": START_TOKEN, "</s>": END_TOKEN,
                        "<pad>": PAD_TOKEN, "<unk>": UNK_TOKEN}
        self.id2sign = dict((idx, token)
                            for token, idx in self.sign2id.items())
        self.length = 4

    def add_sign(self, sign):
        if sign not in self.sign2id:
            self.sign2id[sign] = self.length
            self.id2sign[self.length] = sign
            self.length += 1

    def __len__(self):
        return self.length


def build_vocab(data_dir, min_count=10):
    print("Generating vocabulary...")
    vocab = Vocab()
    counter = Counter()

    formulas_file = join(data_dir, 'im2latex_formulas.norm.lst')
    with open(formulas_file, 'r') as f:
        formulas = [formula.strip('\n') for formula in f.readlines()]

    with open(join(data_dir, 'im2latex_train_filter.lst'), 'r') as f:
        for line in f:
            try:
                _, idx = line.strip('\n').split()
                idx = int(idx)
                formula = formulas[idx].split()
            except:
                pass
            counter.update(formula)

    for word, count in counter.most_common():
        if count >= min_count:
            vocab.add_sign(word)
    vocab_file = join(data_dir, 'vocab.pkl')
    print("Writing vocabulary to ", vocab_file)
    torch.save(vocab, vocab_file)

    print('Vocabulary generated!')


def load_vocab(data_dir):
    print('Loading vocabulary.')
    with open(join(data_dir, 'vocab.pkl'), 'rb') as f:
        vocab = torch.load(f)
    print("Vocabulary loaded! Length: ", len(vocab))
    return vocab


if __name__ == "__main__":
    vocab = build_vocab('processed/')
