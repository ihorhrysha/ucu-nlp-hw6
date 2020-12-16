import nltk
import torch
from sklearn.metrics import precision_recall_fscore_support

# device = torch.device(
#     'cuda') if torch.cuda.is_available() else torch.device('cpu')

# CUDA out of memory so switched to CPU
device = torch.device('cpu')

MODEL_NAME = 'roberta-base'


def split_sentences(text):
    return nltk.sent_tokenize(text)


def flatten_list(list_in):
    return [item for sublist in list_in for item in sublist]


def threshold_labels(probs, threshold):
    labels = [1 if p > threshold else 0 for p in probs]
    return labels


def get_f_score(model, dataset, dataloader, threshold=None):
    pred_labels = model.predict_labels(dataloader, threshold)
    true_labels = dataset.sent_labels
    precision, recall, f_score, _ = \
        precision_recall_fscore_support(flatten_list(true_labels),
                                        flatten_list(pred_labels),
                                        average='binary', beta=0.5)
    return precision, recall, f_score


def get_tp_tn_fp_fn(true_labels, pred_labels):

    comparator_list = [list(a) for a in zip(true_labels, pred_labels)]

    # https://en.wikipedia.org/wiki/F-score
    def f_beta(tp, tn, fp, fn, beta):
        return (1+beta**2)*tp/((1+beta**2)*tp+(beta**2)*fn+fp)

    tp = len([1 for el in comparator_list if el[0] == el[1] and el[1]])

    tn = len([1 for el in comparator_list if el[0] == el[1] and not el[1]])

    fp = len([1 for el in comparator_list if el[0] != el[1] and el[1]])

    fn = len([1 for el in comparator_list if el[0] != el[1] and not el[1]])

    f1 = f_beta(tp, tn, fp, fn, 1)

    f05 = f_beta(tp, tn, fp, fn, .5)

    assert sum([tp, tn, fp, fn]) == len(comparator_list)

    return f05, f1, tp, tn, fp, fn
