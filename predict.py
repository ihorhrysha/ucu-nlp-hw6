import argparse
import torch
from os.path import join
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from dataset import UpperSentDetectorDataset, sent_labels_ids_to_str
from model import UpperSentDetectorModel
from pathlib import Path
from utils import *


def main(args):
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    test_dataset = UpperSentDetectorDataset(tokenizer)
    test_dataset.load(args.input_file)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    model = torch.load(args.model)
    pred_labels = model.predict_labels(test_dataloader, args.threshold)
    out_labels = [sent_labels_ids_to_str(sent_labels)
                  for sent_labels in pred_labels]
    Path(args.output_file).write_text('\n'.join(out_labels))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Path to model')
    parser.add_argument('input_file', help='Input text file with paragraphs')
    parser.add_argument('output_file', help='Output text file')
    parser.add_argument('--batch_size', help='Batch size', type=int, default=4)
    parser.add_argument('--threshold', '-t', help='Threshold', type=float,
                        default=0.90)
    args = parser.parse_args()
    main(args)
