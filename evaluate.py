import argparse
from pathlib import Path
from dataset import sent_labels_str_to_ids
from utils import *


def main(args):
    pred_lines = Path(args.pred_file).read_text().splitlines()
    gold_lines = Path(args.gold_file).read_text().splitlines()
    pred_ids = [sent_labels_str_to_ids(line) for line in pred_lines]
    gold_ids = [sent_labels_str_to_ids(line) for line in gold_lines]
    precision, recall, f_score, _ = \
        precision_recall_fscore_support(flatten_list(gold_ids),
                                        flatten_list(pred_ids),
                                        average='binary', beta=args.beta)
    print('f-score = %1.4f (beta is %1.1f), precision = %1.4f, recall = %1.4f' %
          (f_score, args.beta, precision, recall))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gold_file', help='Output text file')
    parser.add_argument('pred_file', help='File with predictions')
    parser.add_argument('--beta', '-b', help='F-beta value', type=float,
                        default=0.5)
    args = parser.parse_args()
    main(args)
