import argparse
import os

from nltk.translate.bleu_score import corpus_bleu
import torch
from torch import nn, optim
from tqdm import tqdm

# TODO - put this into a function later
from data import *
from models import Seq2seq

dirname = os.path.dirname(os.path.abspath(__file__))


def eval(batch_size=1):
    '''
    TODO - comment
    '''
    model = Seq2seq(SIMPLE_TEXT.vocab, 200)
    model.to(device)
    model.load_state_dict(
        torch.load(
            os.path.join(
                dirname,
                'model.pth'),
            map_location=device))
    model.eval()

    preds = list()
    baselines = list()
    refs = list()

    with torch.no_grad():
        for batch in tqdm(iter(test_iter), total=len(test_iter)):
            for simple, complex in zip(
                    batch.sentence_simple[0].permute(1, 0),
                    batch.sentence_complex[0].permute(1, 0)):
                pred, _ = model.translate_greedy(simple.unsqueeze(1))
                preds.append(pred)
                simple_text = [SIMPLE_TEXT.vocab.itos[tok] for tok in simple]
                complex_text = [SIMPLE_TEXT.vocab.itos[tok] for tok in complex]
                # TODO - get rid of pad
                refs.append([complex_text])
                baselines.append(simple_text)

    print("Model:", corpus_bleu(refs, preds))
    print("Baseline:", corpus_bleu(refs, baselines))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    print('[I] Start eval')
    eval()
    print('[I] Eval finished')
