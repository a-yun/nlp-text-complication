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
    '''
    model.load_state_dict(
        torch.load(
            os.path.join(
                dirname,
                'model.pth'),
            map_location=device))
    '''
    model.eval()

    preds = list()
    refs = list()

    with torch.no_grad():
        for batch in tqdm(iter(test_iter), total=len(test_iter)):
            for simple, complex in zip(
                    batch.sentence_simple[0], batch.sentence_complex[0]):
                pred, _ = model.translate_greedy(simple.unsqueeze(1))
                preds.append(pred)
                refs.append([complex])

    print(corpus_bleu(refs, preds))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    print('[I] Start eval')
    eval()
    print('[I] Eval finished')
