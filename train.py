import argparse
from os.path import join
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from transformers import AdamW
from tqdm import tqdm
from dataset import UpperSentDetectorDataset
from model import UpperSentDetectorModel
from utils import *

import numpy as np


def main(args):
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = UpperSentDetectorDataset(tokenizer)
    train_dataset.load(join(args.data_path, 'train.src'),
                       join(args.data_path, 'train.lbl'))
    test_dataset = UpperSentDetectorDataset(tokenizer)
    test_dataset.load(join(args.data_path, 'test.src'),
                      join(args.data_path, 'test.lbl'))
    dev_dataset = UpperSentDetectorDataset(tokenizer)
    dev_dataset.load(join(args.data_path, 'dev.src'),
                     join(args.data_path, 'dev.lbl'))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size,
                                shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False)

    model = UpperSentDetectorModel(tokenizer)
    model.to(device)

    best_f_score = 0
    optimizer = AdamW(model.parameters(), lr=1e-6)

    for epoch_no in range(args.epoch_num):

        model.train()
        losses = []
        for train_batch in tqdm(train_dataloader,
                                desc='TRAIN epoch # %d' % (epoch_no + 1)):

            # 1) Do forward pass
            outputs = model.forward(train_batch)

            # 2) Calculate loss
            loss_fn = outputs.loss
            losses.append(loss_fn.item())

            # 3) Perform model weights' updates
            loss_fn.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = np.mean(losses)
        print(f'Train loss {avg_train_loss}')

        dev_precision, dev_recall, dev_f_score = get_f_score(
            model, dev_dataset, dev_dataloader)

        print('DEV f05-score = %1.4f, precision = %1.4f, recall = %1.4f | threshold = %1.2f' %
              (dev_f_score, dev_precision, dev_recall, model.threshold))

        if best_f_score > dev_f_score:
            # early stopping
            break
        else:
            best_f_score = dev_f_score

    # Save best on dev model to args.save_model_path
    torch.save(model, args.save_model_path)

    # del model
    model = torch.load(args.save_model_path)

    dev_precision, dev_recall, dev_f_score = get_f_score(model, dev_dataset,
                                                         dev_dataloader)
    test_precision, test_recall, test_f_score = get_f_score(model, test_dataset,
                                                            test_dataloader)
    print('\nFINALLY, FOR BEST MODEL %s:' % 'saved model path')
    print('DEV f05-score = %1.4f, precision = %1.4f, recall = %1.4f | threshold = %1.2f' %
          (dev_f_score, dev_precision, dev_recall, model.threshold))
    print('TEST f05-score = %1.4f, precision = %1.4f, recall = %1.4f | threshold = %1.2f' %
          (test_f_score, test_precision, test_recall, model.threshold))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='Path with train/dev/test files',
                        default='data/')
    parser.add_argument('--save_model_path', '-m',
                        help='Path to saved model')
    parser.add_argument('--epoch_num', '-e', type=int, default=5,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    args = parser.parse_args()

    main(args)
