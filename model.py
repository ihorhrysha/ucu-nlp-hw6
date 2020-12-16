from transformers import RobertaForTokenClassification
from tqdm import tqdm
from utils import *


class UpperSentDetectorModel(torch.nn.Module):
    def __init__(self, tokenizer):
        super(UpperSentDetectorModel, self).__init__()
        self.tokenizer = tokenizer
        self.transformer = \
            RobertaForTokenClassification.from_pretrained('roberta-base')
        self.transformer.to(device)
        self.softmax = torch.nn.Softmax(dim=2)
        self.threshold = 0.9  # may be changed, if you need i

    def forward(self, batch):
        # make forward step on batch
        outputs = self.transformer(
            input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch.get('token_labels'))

        return outputs

    def predict_probs(self, dataloader):
        self.eval()
        probs = []
        for batch in tqdm(dataloader, desc='INFERENCE'):
            outputs = self.forward(batch)
            logits = outputs.logits
            batch_probs = self.softmax(logits)[:, :, 1]

            # Convert token-level probs to sentence-level probs
            token_labels = batch['token_labels']

            for i in range(len(token_labels)):
                mask = token_labels[i] >= 0
                probs.append(batch_probs[i][mask].tolist())

        return probs

    def predict_labels(self, dataloader, threshold=None):
        pred_probs = self.predict_probs(dataloader)
        if threshold is None:
            threshold = self.threshold
        pred_labels = [threshold_labels(curr_probs, threshold)
                       for curr_probs in pred_probs]
        return pred_labels
