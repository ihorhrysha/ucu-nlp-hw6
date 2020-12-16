import copy
import torch
from torch.utils.data import Dataset
from pathlib import Path
from os import path
from utils import *


def sent_labels_str_to_ids(sent_labels_str):
    return [1 if s == 'C' else 0 for s in sent_labels_str.split()]


def sent_labels_ids_to_str(sent_labels_ids):
    return ' '.join(['C' if idx == 1 else 'O' for idx in sent_labels_ids])


class UpperSentDetectorDataset(Dataset):
    def __init__(self, tokenizer):
        super(Dataset, self).__init__()
        self.tokenizer = tokenizer
        self.encodings = []
        self.token_labels = []
        self.sent_labels = []
        self.num_lines = 0

    def add_cls_to_line(self, line):
        sentences = split_sentences(line)

        # Add code which adds [CLS] token to the end of each sentence.
        return self.tokenizer.cls_token.join(sentences) + self.tokenizer.cls_token

    def tokenize(self, lines):
        # Avoid adding spec tokens especially cls at the begining
        return self.tokenizer(lines, truncation=True, padding=True, add_special_tokens=False)

    def load(self, lines_filename, sent_labels_filename=None, verbose=True):

        # Load source paragraphs
        if not path.isfile(lines_filename):
            raise FileNotFoundError()

        with open(lines_filename, 'r') as f:
            lines = [self.add_cls_to_line(line) for line in f.readlines()]

            self.num_lines = len(lines)

            if verbose:
                print(f'Data items loaded: {len(lines)}')

            # Tokenize them using tokenizer
            self.encodings = self.tokenize(lines)

        # Load sentence-level labels and convert them to token-level labels.
        if sent_labels_filename:

            if not path.isfile(sent_labels_filename):
                raise FileNotFoundError()

            with open(sent_labels_filename, 'r') as f:

                # lines to sent level labels token list
                self.sent_labels = [sent_labels_str_to_ids(
                    line.strip()) for line in f.readlines()]

                assert self.num_lines == len(
                    self.sent_labels), f'Length of examples {self.num_lines} and labels {len(self.sent_labels)} should be equal!'

            if verbose:
                print(f'Labels loaded loaded: {self.num_lines}')
        else:

            self.sent_labels = []

        input_ids = self.encodings['input_ids']

        for i in range(self.num_lines):

            if len(self.sent_labels):
                # [O, C, O] to match with cls_token
                sent_labels_id = copy.copy(self.sent_labels[i])
            else:
                # just zeros for mask
                sent_labels_id = [0]*100

            line_token_labels = [sent_labels_id.pop(0) if token ==
                                 self.tokenizer.cls_token_id else - 100 for token in input_ids[i]]

            self.token_labels.append(line_token_labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to(device) for key, val in
                self.encodings.items()}
        if self.token_labels is not None:
            item['token_labels'] = \
                torch.tensor(self.token_labels[idx]).to(device)
        return item

    def __len__(self):
        return self.num_lines
